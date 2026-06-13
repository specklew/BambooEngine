#include "pch.h"
#include "VoxelizationPass.h"

#include <algorithm>
#include <cstring>

#include "InputElements.h"
#include "Shader.h"
#include "ResourceManager/ResourceManager.h"
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
    constexpr uint32_t kCbAlignedSize = 256;

    uint32_t Align(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }
}

void VoxelizationPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    ComPtr<ID3D12RootSignature>        rasterRootSignature)
{
    spdlog::info("Initializing voxelization pass...");

    m_device              = device;
    m_commandList         = commandList;
    m_rasterRootSignature = rasterRootSignature;

    D3D12_FEATURE_DATA_D3D12_OPTIONS opts = {};
    if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts))))
    {
        if (opts.ConservativeRasterizationTier < D3D12_CONSERVATIVE_RASTERIZATION_TIER_1)
            spdlog::warn("Conservative rasterization unsupported on this device! Voxelization may miss thin triangles.");
    }

    CreateResources();
    CreateDescriptorHeap();
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

    // Grid constants CB (upload heap, persistently mapped)
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

void VoxelizationPass::CreateDescriptorHeap()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 3; // occupancy + irradiance + vpl count
    heapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_descHeap)));

    UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_descHeap->GetCPUDescriptorHandleForHeapStart());

    WriteUintTex3DUav(m_occupancyTex.Get(), handle);
    handle.Offset(1, inc);
    WriteUintTex3DUav(m_irradianceTex.Get(), handle);
    handle.Offset(1, inc);
    WriteUintTex3DUav(m_vplCountTex.Get(), handle);
}

void VoxelizationPass::CreateRootSignatures()
{
    // Clear root sig: root constants b0 (gGridDim), descriptor table of 3 UAVs (u0..u2)
    {
        CD3DX12_DESCRIPTOR_RANGE uavRange;
        uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0);

        CD3DX12_ROOT_PARAMETER params[2];
        params[0].InitAsConstants(4, 0); // 4 uints at b0
        params[1].InitAsDescriptorTable(1, &uavRange);

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> sig, err;
        ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
        ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
            IID_PPV_ARGS(&m_clearRootSig)));
        m_clearRootSig->SetName(L"VoxelClear RootSig");
    }

    // Voxelize root sig: CBV b0 (grid), CBV b1 (model), root constant b2 (axis), descriptor table u0
    {
        CD3DX12_DESCRIPTOR_RANGE uavRange;
        uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

        CD3DX12_ROOT_PARAMETER params[4];
        params[0].InitAsConstantBufferView(0);            // b0 grid
        params[1].InitAsConstantBufferView(1);            // b1 model
        params[2].InitAsConstants(4, 2);                  // b2 axis (4 uints)
        params[3].InitAsDescriptorTable(1, &uavRange);    // u0 occupancy

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
        if (err) OutputDebugStringA(static_cast<char*>(err->GetBufferPointer()));
        ThrowIfFailed(hr);
        ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
            IID_PPV_ARGS(&m_voxelizeRootSig)));
        m_voxelizeRootSig->SetName(L"Voxelize RootSig");
    }
}

void VoxelizationPass::CreatePSOs()
{
    auto& rm = ResourceManager::Get();

    // Clear PSO
    {
        auto csh = rm.GetOrLoadShader(AssetId("resources/shaders/clearVoxels.cs.shader"));
        auto csBlob = rm.shaders.GetResource(csh).bytecode;

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_clearRootSig.Get();
        desc.CS = CD3DX12_SHADER_BYTECODE(csBlob->GetBufferPointer(), csBlob->GetBufferSize());
        ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_clearPso)));
        m_clearPso->SetName(L"VoxelClear PSO");
    }

    // Voxelize PSO (graphics, UAV-only)
    {
        auto vsh = rm.GetOrLoadShader(AssetId("resources/shaders/voxelize.vs.shader"));
        auto psh = rm.GetOrLoadShader(AssetId("resources/shaders/voxelize.ps.shader"));
        auto vsBlob = rm.shaders.GetResource(vsh).bytecode;
        auto psBlob = rm.shaders.GetResource(psh).bytecode;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_voxelizeRootSig.Get();
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

        ThrowIfFailed(m_device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_voxelizePso)));
        m_voxelizePso->SetName(L"Voxelize PSO");
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
    m_descHeap.Reset();

    CreateResources();
    CreateDescriptorHeap();
    m_didResize = true;
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

void VoxelizationPass::DispatchClear()
{
    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_clearRootSig.Get());
    m_commandList->SetPipelineState(m_clearPso.Get());

    uint32_t clearParams[4] = { m_gridDim, 0, 0, 0 };
    m_commandList->SetComputeRoot32BitConstants(0, 4, clearParams, 0);
    m_commandList->SetComputeRootDescriptorTable(1, m_descHeap->GetGPUDescriptorHandleForHeapStart());

    const uint32_t groups = (m_gridDim + 7) / 8;
    m_commandList->Dispatch(groups, groups, groups);

    D3D12_RESOURCE_BARRIER barriers[3];
    barriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(m_occupancyTex.Get());
    barriers[1] = CD3DX12_RESOURCE_BARRIER::UAV(m_irradianceTex.Get());
    barriers[2] = CD3DX12_RESOURCE_BARRIER::UAV(m_vplCountTex.Get());
    m_commandList->ResourceBarrier(_countof(barriers), barriers);
}

void VoxelizationPass::DispatchVoxelize(const Scene& scene)
{
    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetGraphicsRootSignature(m_voxelizeRootSig.Get());
    m_commandList->SetPipelineState(m_voxelizePso.Get());

    D3D12_VIEWPORT viewport = { 0.0f, 0.0f, static_cast<float>(m_gridDim), static_cast<float>(m_gridDim), 0.0f, 1.0f };
    D3D12_RECT     scissor  = { 0, 0, static_cast<LONG>(m_gridDim), static_cast<LONG>(m_gridDim) };
    m_commandList->RSSetViewports(1, &viewport);
    m_commandList->RSSetScissorRects(1, &scissor);
    m_commandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);

    m_commandList->SetGraphicsRootConstantBufferView(0, m_gridConstantsCB->GetGPUVirtualAddress());
    m_commandList->SetGraphicsRootDescriptorTable(3, m_descHeap->GetGPUDescriptorHandleForHeapStart());

    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        uint32_t axisParams[4] = { axis, 0, 0, 0 };
        m_commandList->SetGraphicsRoot32BitConstants(2, 4, axisParams, 0);

        for (const auto& go : scene.GetGameObjects())
        {
            auto worldGpu = go->GetWorldMatrixBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress();
            m_commandList->SetGraphicsRootConstantBufferView(1, worldGpu);

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

                m_commandList->DrawIndexedInstanced(
                    static_cast<UINT>(index_view.count), 1,
                    static_cast<UINT>(index_view.offset),
                    static_cast<INT>(vertex_view.offset), 0);
            }
        }
    }

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_occupancyTex.Get());
    m_commandList->ResourceBarrier(1, &barrier);
}

void VoxelizationPass::RunFrame(const Scene& scene, uint32_t requestedGridDim)
{
    if (!m_initialized || !m_haveScene) return;

    m_didResize = false;
    if (requestedGridDim != m_gridDim)
    {
        RecreateForNewDim(requestedGridDim);
        OnSceneLoaded(scene);
    }

    DispatchClear();
    DispatchVoxelize(scene);
}
