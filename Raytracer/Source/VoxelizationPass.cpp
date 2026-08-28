#include "pch.h"
#include "CommandContext.h"
#include "VoxelizationPass.h"

#include <algorithm>
#include <cstring>

#include "Constants.h"
#include "InputElements.h"
#include "Shader.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "ShaderReflection.h"
#include "Resources/ConstantBuffer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/Scene.h"
#include "SceneResources/GameObject.h"
#include "SceneResources/Model.h"
#include "SceneResources/Primitive.h"

using Microsoft::WRL::ComPtr;

namespace
{
// Three signatures: the per-frame accumulator clear, the bake-time clear, and
// the bake draw itself. All three address the three voxel textures at the global
// heap slots the renderer writes when the grid is created or resized.
constexpr BindingSlot kFrameClearConstants = PassRootConstants("ClearCB", VOXEL_FRAME_CLEAR_REG_CB, 4);
constexpr BindingSlot kFrameClearIrradiance =
    PassTableEntry("gIrradiance", BindingKind::Uav, VOXEL_FRAME_CLEAR_REG_IRRADIANCE, GlobalDescriptor::VoxelIrradiance);
constexpr BindingSlot kFrameClearVplCount =
    PassTableEntry("gVplCount", BindingKind::Uav, VOXEL_FRAME_CLEAR_REG_VPL_COUNT, GlobalDescriptor::VoxelVplCount);
constexpr BindingSlot kFrameClearSlots[] = {kFrameClearConstants, kFrameClearIrradiance, kFrameClearVplCount};

constexpr BindingSlot kBakeClearConstants = PassRootConstants("ClearCB", VOXEL_BAKE_CLEAR_REG_CB, 4);
constexpr BindingSlot kBakeClearOccupancy =
    PassTableEntry("gOccupancy", BindingKind::Uav, VOXEL_BAKE_CLEAR_REG_OCCUPANCY, GlobalDescriptor::VoxelOccupancy);
constexpr BindingSlot kBakeClearBoundMin = PassUav("gBakedBoundMin", VOXEL_BAKE_CLEAR_REG_BAKED_BOUND_MIN);
constexpr BindingSlot kBakeClearBoundMax = PassUav("gBakedBoundMax", VOXEL_BAKE_CLEAR_REG_BAKED_BOUND_MAX);
constexpr BindingSlot kBakeClearSlots[] = {kBakeClearConstants, kBakeClearOccupancy, kBakeClearBoundMin,
                                           kBakeClearBoundMax};

constexpr BindingSlot kBakeGridConstants  = PassCbv("VoxelGridCB", VOXEL_BAKE_REG_GRID_CB);
constexpr BindingSlot kBakeModelConstants = PassCbv("ModelTransforms", VOXEL_BAKE_REG_MODEL_CB);
constexpr BindingSlot kBakeAxisConstants  = PassRootConstants("BakeCB", VOXEL_BAKE_REG_AXIS_CB, 4); // axis + bound flags
constexpr BindingSlot kBakeOccupancy =
    PassTableEntry("gOccupancy", BindingKind::Uav, VOXEL_BAKE_REG_OCCUPANCY, GlobalDescriptor::VoxelOccupancy);
constexpr BindingSlot kBakeBoundMin  = PassUav("gBakedBoundMin", VOXEL_BAKE_REG_BAKED_BOUND_MIN);
constexpr BindingSlot kBakeBoundMax  = PassUav("gBakedBoundMax", VOXEL_BAKE_REG_BAKED_BOUND_MAX);
constexpr BindingSlot kBakeSlots[] = {kBakeGridConstants, kBakeModelConstants, kBakeAxisConstants, kBakeOccupancy,
                                      kBakeBoundMin,      kBakeBoundMax};

    constexpr uint32_t kCbAlignedSize = 256;

    uint32_t Align(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }
}

void VoxelizationPass::Initialize(ComPtr<ID3D12Device5> device, ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    spdlog::info("Initializing voxelization pass...");

    m_device      = device;
    m_commandList = commandList;

    D3D12_FEATURE_DATA_D3D12_OPTIONS opts = {};
    if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts))))
    {
        if (opts.ConservativeRasterizationTier < D3D12_CONSERVATIVE_RASTERIZATION_TIER_1)
            spdlog::warn("Conservative rasterization unsupported on this device! Voxelization may miss thin triangles.");
    }

    CreateResources();
    CreateRootSignatures();
    CreatePSOs();

    m_initialized = true;
    spdlog::info("Voxelization pass initialized: gridDim={}", m_gridDim);
}

void VoxelizationPass::CreateResources()
{
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    // Occupancy Texture3D<uint>
    {
        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex3D(
            DXGI_FORMAT_R32_UINT, m_gridDim, m_gridDim, m_gridDim, 1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_occupancyTex)));
        m_occupancyTex->SetName(L"VoxelOccupancy");
    }

    // Packed irradiance + VPL count (Texture3D<uint>, uint atomics from injection pass)
    {
        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex3D(
            DXGI_FORMAT_R32_UINT, m_gridDim, m_gridDim, m_gridDim, 1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_irradianceTex)));
        m_irradianceTex->SetName(L"VoxelIrradiance");

        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_vplCountTex)));
        m_vplCountTex->SetName(L"VoxelVplCount");
    }

    // Baked per-voxel bounds: 4 uints per cell, quantized to the voxel cube.
    {
        const uint32_t cellCount = m_gridDim * m_gridDim * m_gridDim;
        m_bakedBoundMin = std::make_unique<RWStructuredBuffer<uint32_t>>(
            m_device, static_cast<size_t>(cellCount) * 4, L"VoxelBakedBoundMin");
        m_bakedBoundMax = std::make_unique<RWStructuredBuffer<uint32_t>>(
            m_device, static_cast<size_t>(cellCount) * 4, L"VoxelBakedBoundMax");
    }

    // Grid constants CB (upload heap, persistently mapped)
    if (!m_gridConstantsCB)
    {
        auto uploadProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto cbDesc      = CD3DX12_RESOURCE_DESC::Buffer(kCbAlignedSize);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &uploadProps, D3D12_HEAP_FLAG_NONE, &cbDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_gridConstantsCB)));
        m_gridConstantsCB->SetName(L"VoxelGridConstants CB");
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(m_gridConstantsCB->Map(0, &readRange, &m_gridConstantsCBMapped));
    }
}

void VoxelizationPass::CreateRootSignatures()
{
    m_clearRootSig = RootSignatureBuilder(L"VoxelFrameClear RootSig", /*tableCount*/ 1)
                         .Add(kFrameClearSlots)
                         .Build(m_device.Get());

    m_bakeClearRootSig = RootSignatureBuilder(L"VoxelBakeClear RootSig", /*tableCount*/ 1)
                             .Add(kBakeClearSlots)
                             .Build(m_device.Get());

    // The bake is a conservative-raster draw, so its binds go through
    // SetGraphicsRoot* and it needs the input-assembler flag.
    m_bakeRootSig = RootSignatureBuilder(L"VoxelBake RootSig", /*tableCount*/ 1)
                        .ForGraphics()
                        .WithFlags(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)
                        .Add(kBakeSlots)
                        .Build(m_device.Get());
}

void VoxelizationPass::CreatePSOs()
{
    auto& rm = ResourceManager::Get();

    auto& cache = ShaderProgramCache::Get();

    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_clearRootSig.Get(),
        "resources/shaders/clearVoxels.cs.shader", L"VoxelFrameClear PSO");
    m_bakeClearProgram = cache.GetOrCreateCompute(m_device.Get(), m_bakeClearRootSig.Get(),
        "resources/shaders/bakeClear.cs.shader", L"VoxelBakeClear PSO");

    // Bake PSO (graphics, UAV-only)
    {
        auto vsh = rm.GetOrLoadShader(AssetId("resources/shaders/voxelize.vs.shader"));
        auto psh = rm.GetOrLoadShader(AssetId("resources/shaders/voxelize.ps.shader"));
        auto vsBlob = rm.shaders.GetResource(vsh).bytecode;
        auto psBlob = rm.shaders.GetResource(psh).bytecode;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_bakeRootSig.Get();
        desc.VS = { static_cast<BYTE*>(vsBlob->GetBufferPointer()), vsBlob->GetBufferSize() };
        desc.PS = { static_cast<BYTE*>(psBlob->GetBufferPointer()), psBlob->GetBufferSize() };
        desc.InputLayout = { inputLayout, _countof(inputLayout) };

        CD3DX12_RASTERIZER_DESC rasterDesc(D3D12_DEFAULT);
        rasterDesc.CullMode                 = D3D12_CULL_MODE_NONE;
        rasterDesc.ConservativeRaster       = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
        rasterDesc.DepthClipEnable          = FALSE;
        desc.RasterizerState                = rasterDesc;

        CD3DX12_DEPTH_STENCIL_DESC dsDesc(D3D12_DEFAULT);
        dsDesc.DepthEnable      = FALSE;
        dsDesc.StencilEnable    = FALSE;
        desc.DepthStencilState  = dsDesc;
        desc.DSVFormat          = DXGI_FORMAT_UNKNOWN;

        desc.BlendState              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.SampleMask              = UINT_MAX;
        desc.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets        = 0;
        desc.SampleDesc.Count        = 1;

        ThrowIfFailed(m_device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_bakePso)));
        m_bakePso->SetName(L"VoxelBake PSO");

        ShaderReflection::ValidateShaderAsset("resources/shaders/voxelize.vs.shader", m_bakeRootSig.Get());
        ShaderReflection::ValidateShaderAsset("resources/shaders/voxelize.ps.shader", m_bakeRootSig.Get());
    }
}

void VoxelizationPass::WriteUintTex3DUav(ID3D12Resource* resource, D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    if (!resource) return;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format          = DXGI_FORMAT_R32_UINT;
    uavDesc.ViewDimension   = D3D12_UAV_DIMENSION_TEXTURE3D;
    uavDesc.Texture3D.WSize = m_gridDim;
    m_device->CreateUnorderedAccessView(resource, nullptr, &uavDesc, dest);
}

void VoxelizationPass::WriteOccupancyUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    WriteUintTex3DUav(m_occupancyTex.Get(), dest);
}

void VoxelizationPass::WriteIrradianceUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    WriteUintTex3DUav(m_irradianceTex.Get(), dest);
}

void VoxelizationPass::WriteVplCountUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    WriteUintTex3DUav(m_vplCountTex.Get(), dest);
}

void VoxelizationPass::SetRuntimeParams(bool injectUseAvg, float heatScale)
{
    m_gridConstants.injectUseAvg = injectUseAvg ? 1u : 0u;
    m_gridConstants.heatScale    = heatScale;
    WriteGridConstantsCB();
}

void VoxelizationPass::RecreateForNewDim(uint32_t newDim)
{
    newDim = std::clamp(newDim, 4u, 512u);
    if (newDim == m_gridDim) return;

    spdlog::debug("VoxelizationPass: resizing grid {} -> {}", m_gridDim, newDim);
    m_gridDim = newDim;

    m_occupancyTex.Reset();
    m_irradianceTex.Reset();
    m_vplCountTex.Reset();
    m_bakedBoundMin.reset();
    m_bakedBoundMax.reset();

    CreateResources();
    m_didResize = true;
    m_bakeValid = false;
}

void VoxelizationPass::OnSceneLoaded(const Scene& scene)
{
    const DirectX::XMFLOAT3& aabbMin = scene.GetAabbMin();
    const DirectX::XMFLOAT3& aabbMax = scene.GetAabbMax();

    m_cachedAabbMin = aabbMin;
    m_cachedAabbMax = aabbMax;

    DirectX::XMFLOAT3 size{
        std::max(aabbMax.x - aabbMin.x, 1e-4f),
        std::max(aabbMax.y - aabbMin.y, 1e-4f),
        std::max(aabbMax.z - aabbMin.z, 1e-4f),
    };
    const float maxExtent = std::max({ size.x, size.y, size.z });
    const float voxelSize = maxExtent / static_cast<float>(m_gridDim);

    DirectX::XMFLOAT3 center{
        (aabbMin.x + aabbMax.x) * 0.5f,
        (aabbMin.y + aabbMax.y) * 0.5f,
        (aabbMin.z + aabbMax.z) * 0.5f,
    };
    const float half = voxelSize * static_cast<float>(m_gridDim) * 0.5f;

    m_gridConstants.gridMin   = { center.x - half, center.y - half, center.z - half };
    m_gridConstants.gridMax   = { center.x + half, center.y + half, center.z + half };
    m_gridConstants.voxelSize = voxelSize;
    m_gridConstants.gridDim   = m_gridDim;

    WriteGridConstantsCB();
    m_haveScene = true;
    m_bakeValid = false;

    spdlog::debug("VoxelizationPass: gridMin=({:.3f},{:.3f},{:.3f}) gridMax=({:.3f},{:.3f},{:.3f}) voxelSize={:.4f}",
        m_gridConstants.gridMin.x, m_gridConstants.gridMin.y, m_gridConstants.gridMin.z,
        m_gridConstants.gridMax.x, m_gridConstants.gridMax.y, m_gridConstants.gridMax.z,
        m_gridConstants.voxelSize);
}

void VoxelizationPass::WriteGridConstantsCB()
{
    if (!m_gridConstantsCBMapped) return;
    std::memcpy(m_gridConstantsCBMapped, &m_gridConstants, sizeof(VoxelGridConstants));
}

void VoxelizationPass::DispatchFrameClear()
{
    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_clearRootSig.Get());
    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());

    uint32_t clearParams[4] = { m_gridDim, 0, 0, 0 };
    m_clearRootSig.SetConstants(m_commandList.Get(), kFrameClearConstants, clearParams, 4);
    m_clearRootSig.SetTable(m_commandList.Get(), 0, globalHeap.GpuStart());

    const uint32_t groups = (m_gridDim + 7) / 8;
    CommandContext::Get().Dispatch(groups, groups, groups);
}

void VoxelizationPass::DispatchBakeClear()
{
    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_bakeClearRootSig.Get());
    m_commandList->SetPipelineState(m_bakeClearProgram->GetPipelineState());

    uint32_t clearParams[4] = { m_gridDim, 0, 0, 0 };
    m_bakeClearRootSig.SetConstants(m_commandList.Get(), kBakeClearConstants, clearParams, 4);
    m_bakeClearRootSig.SetTable(m_commandList.Get(), 0, globalHeap.GpuStart());
    m_bakeClearRootSig.Set(m_commandList.Get(), kBakeClearBoundMin, m_bakedBoundMin->GetGPUVirtualAddress());
    m_bakeClearRootSig.Set(m_commandList.Get(), kBakeClearBoundMax, m_bakedBoundMax->GetGPUVirtualAddress());

    const uint32_t groups = (m_gridDim + 7) / 8;
    CommandContext::Get().Dispatch(groups, groups, groups);
}

void VoxelizationPass::DispatchBake(const Scene& scene)
{
    spdlog::info("VoxelizationPass: baking geometry (gridDim={}, useCompact={}, clipping={})",
        m_gridDim, m_bakedUseCompact, m_bakedClipping);

    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetGraphicsRootSignature(m_bakeRootSig.Get());
    m_commandList->SetPipelineState(m_bakePso.Get());

    D3D12_VIEWPORT viewport = { 0.0f, 0.0f, static_cast<float>(m_gridDim), static_cast<float>(m_gridDim), 0.0f, 1.0f };
    D3D12_RECT     scissor  = { 0, 0, static_cast<LONG>(m_gridDim), static_cast<LONG>(m_gridDim) };
    m_commandList->RSSetViewports(1, &viewport);
    m_commandList->RSSetScissorRects(1, &scissor);
    m_commandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);

    m_bakeRootSig.Set(m_commandList.Get(), kBakeGridConstants, m_gridConstantsCB->GetGPUVirtualAddress());
    m_bakeRootSig.SetTable(m_commandList.Get(), 0, globalHeap.GpuStart());
    m_bakeRootSig.Set(m_commandList.Get(), kBakeBoundMin, m_bakedBoundMin->GetGPUVirtualAddress());
    m_bakeRootSig.Set(m_commandList.Get(), kBakeBoundMax, m_bakedBoundMax->GetGPUVirtualAddress());

    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        uint32_t bakeParams[4] = {
            axis,
            m_bakedUseCompact ? 1u : 0u,
            m_bakedClipping ? 1u : 0u,
            0,
        };
        m_bakeRootSig.SetConstants(m_commandList.Get(), kBakeAxisConstants, bakeParams, 4);

        for (const auto& go : scene.GetGameObjects())
        {
            auto worldGpu = go->GetWorldMatrixBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress();
            m_bakeRootSig.Set(m_commandList.Get(), kBakeModelConstants, worldGpu);

            for (const auto& primitive : go->GetModel()->GetMeshes())
            {
                auto vertex_view = primitive->GetVertexView();
                auto index_view  = primitive->GetIndexView();

                auto vb = std::dynamic_pointer_cast<VertexBuffer>(vertex_view.buffer);
                auto ib = std::dynamic_pointer_cast<IndexBuffer>(index_view.buffer);
                if (!vb || !ib) continue;

                m_commandList->IASetVertexBuffers(0, 1, &vb->GetVertexBufferView());
                m_commandList->IASetIndexBuffer(&ib->GetIndexBufferView());
                m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                CommandContext::Get().DrawIndexedInstanced(
                    static_cast<UINT>(index_view.count), 1,
                    static_cast<UINT>(index_view.offset),
                    static_cast<INT>(vertex_view.offset), 0);
            }
        }
    }

    m_bakeValid = true;
}

bool VoxelizationPass::PrepareFrame(const Scene& scene, uint32_t requestedGridDim, bool bakeUseCompact, bool bakeClipping)
{
    if (!m_initialized || !m_haveScene) return false;

    m_didResize = false;
    if (requestedGridDim != m_gridDim)
    {
        RecreateForNewDim(requestedGridDim);
        OnSceneLoaded(scene);
    }

    if (bakeUseCompact != m_bakedUseCompact || bakeClipping != m_bakedClipping)
    {
        m_bakedUseCompact = bakeUseCompact;
        m_bakedClipping   = bakeClipping;
        m_bakeValid       = false;
    }

    return !m_bakeValid;
}
