#include "pch.h"
#include "CommandContext.h"
#include "VxpgClusterVisibilityPass.h"

#include "Constants.h"
#include "GlobalDescriptorHeap.h"
#include "Renderer.h" // GetStaticSamplers
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgClusterPass.h"
#include "SuperpixelBuildPass.h"
#include "SceneResources/Scene.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/CVars.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

// SIByL cvis defaults: use_bsdf = true, use_distance = false (BRDF-weighted soft
// visibility). Both exposed so the weighting can be toggled for measurement.
static AutoCVarInt g_cvisUseBsdf("vxpg.cvis.useBsdf",
    "Weight cluster visibility by the receiver Cook-Torrance BRDF (SIByL default on)",
    1, CVarFlags::EditCheckbox);
static AutoCVarInt g_cvisUseDistance("vxpg.cvis.useDistance",
    "Also weight cluster visibility by inverse-square distance (SIByL default off)",
    0, CVarFlags::EditCheckbox);

namespace
{
    constexpr uint32_t kClusterCount = 32;
    constexpr uint32_t kGatherCap = 1024;
    constexpr uint32_t kSuperpixelSize = Constants::Graphics::SUPERPIXEL_SIZE;
}

void VxpgClusterVisibilityPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgClusterPass>       clusterPass,
    std::shared_ptr<SuperpixelBuildPass>   superpixelPass)
{
    spdlog::info("Initializing VXPG cluster-visibility pass...");

    m_device         = device;
    m_commandList    = commandList;
    m_voxelPass      = std::move(voxelPass);
    m_buildPass      = std::move(buildPass);
    m_clusterPass    = std::move(clusterPass);
    m_superpixelPass = std::move(superpixelPass);

    CreateFixedBuffers();
    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void VxpgClusterVisibilityPass::CreateFixedBuffers()
{
    m_clusterGatheredLightPoints = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT4>>(
        m_device, kClusterCount * kGatherCap, L"ClusterVisibility GatheredLightPoints");
    m_clusterLightPointCounts = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, kClusterCount, L"ClusterVisibility LightPointCounts");
}

void VxpgClusterVisibilityPass::CreateResolutionBuffers()
{
    m_avgVisibility = std::make_unique<RWStructuredBuffer<float>>(
        m_device, std::max(1u, m_mapX * m_mapY * kClusterCount), L"ClusterVisibility AvgVisibility");

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32_UINT, std::max(1u, m_mapX), std::max(1u, m_mapY),
        1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_mask)));
    m_mask->SetName(L"ClusterVisibility Mask");
}

void VxpgClusterVisibilityPass::CreateRootSignature()
{
    // No texture range: the BSDF weight uses per-instance material factors, so
    // this compute pass never samples the scene textures (which sit in the
    // raster path's PIXEL_SHADER_RESOURCE layout, illegal for a compute Dispatch).
    // It includes RaytracingUtils.hlsl, so the scene half of its bindings is the
    // frame layout; only the cvis-specific UAVs are pass-scoped.
    CD3DX12_DESCRIPTOR_RANGE r[10];
    r[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, FRAME_REG_CAMERA_MATRICES, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::CameraMatrices));
    r[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, FRAME_REG_TLAS, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Tlas));
    r[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, FRAME_REG_VERTICES, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Vertices));
    r[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, FRAME_REG_INDICES, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Indices));
    r[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, CVIS_REG_SUPERPIXEL_INDEX, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::SuperpixelIndex));
    r[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, CVIS_REG_VPL_POSITION, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VplPosition));
    r[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, CVIS_REG_VBUFFER, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VBuffer));
    r[7].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, CVIS_REG_SPIXEL_GATHERED, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::SpixelGathered));
    r[8].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, CVIS_REG_SPIXEL_COUNTER, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::SpixelCounter));
    r[9].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, CVIS_REG_VISIBILITY_MASK, 0, GlobalDescriptorHeap::IndexOf(GlobalDescriptor::ClusterVisibilityMask));

    CD3DX12_ROOT_PARAMETER params[10];
    params[0].InitAsDescriptorTable(_countof(r), r);
    params[1].InitAsShaderResourceView(FRAME_REG_GEOMETRY_INFO);
    params[2].InitAsShaderResourceView(FRAME_REG_INSTANCE_INFO);
    params[3].InitAsConstantBufferView(CVIS_REG_GRID_CB);
    params[4].InitAsConstants(8, CVIS_REG_CB);
    params[5].InitAsUnorderedAccessView(CVIS_REG_INVERSE_INDEX);
    params[6].InitAsUnorderedAccessView(CVIS_REG_CLUSTER_ASSIGNMENTS);
    params[7].InitAsUnorderedAccessView(CVIS_REG_GATHERED_LIGHT_POINTS);
    params[8].InitAsUnorderedAccessView(CVIS_REG_LIGHT_POINT_COUNTS);
    params[9].InitAsUnorderedAccessView(CVIS_REG_AVG_VISIBILITY);

    auto samplers = Renderer::GetStaticSamplers();
    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params,
        static_cast<UINT>(samplers.size()), samplers.data(),
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    m_rootSig = RootSignatureLibrary::Get().Create(m_device.Get(), desc, L"VxpgClusterVisibility RootSig");
}

void VxpgClusterVisibilityPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgClusterVisibility.clear.shader", L"Cvis Clear PSO");
    m_gatherProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgClusterVisibility.gather.shader", L"Cvis Gather PSO");
    m_checkProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgClusterVisibility.check.shader", L"Cvis Check PSO");
}

void VxpgClusterVisibilityPass::OnResize(uint32_t width, uint32_t height)
{
    m_width  = width;
    m_height = height;
    m_mapX   = (width  + kSuperpixelSize - 1) / kSuperpixelSize;
    m_mapY   = (height + kSuperpixelSize - 1) / kSuperpixelSize;
    CreateResolutionBuffers();
}

bool VxpgClusterVisibilityPass::BindCommon(uint32_t frameIndex)
{
    if (!m_initialized || !m_scene || !m_voxelPass || !m_buildPass ||
        !m_clusterPass || !m_superpixelPass || m_mapX == 0)
        return false;

    auto* cmd = m_commandList.Get();

    ID3D12DescriptorHeap* heaps[] = { GlobalDescriptorHeap::Get().GetHeap() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    cmd->SetComputeRootSignature(m_rootSig.Get());
    cmd->SetComputeRootDescriptorTable(0, GlobalDescriptorHeap::Get().GpuStart());
    cmd->SetComputeRootShaderResourceView(1, m_scene->GetGeometryInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    cmd->SetComputeRootShaderResourceView(2, m_scene->GetInstanceInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    cmd->SetComputeRootConstantBufferView(3, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());

    uint32_t constants[8] = {
        m_width, m_height, m_mapX, m_mapY, frameIndex,
        static_cast<uint32_t>(g_cvisUseBsdf.Get() != 0),
        static_cast<uint32_t>(g_cvisUseDistance.Get() != 0),
        m_scene->GetInstanceInfoBuffer()->GetElementsCount()
    };
    cmd->SetComputeRoot32BitConstants(4, 8, constants, 0);
    cmd->SetComputeRootUnorderedAccessView(5, m_buildPass->GetInverseIndexBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(6, m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(7, m_clusterGatheredLightPoints->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(8, m_clusterLightPointCounts->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(9, m_avgVisibility->GetGPUVirtualAddress());

    return true;
}

void VxpgClusterVisibilityPass::RunClear(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_mapX + 15) / 16, (m_mapY + 15) / 16, 1);
}

// File each pixel's VPL into its cluster drawer and seed the mask bits for the
// connections it already proves.
void VxpgClusterVisibilityPass::RunGather(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_gatherProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);
}

// Shadow-ray probes -> soft avg-visibility + mask. The dispatch covers mapX
// superpixels wide (32 sample lanes each) and mapY*4 groups tall (8 clusters per
// group x 4 = 32 clusters per superpixel row).
void VxpgClusterVisibilityPass::RunCheck(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_checkProgram->GetPipelineState());
    CommandContext::Get().Dispatch(m_mapX, m_mapY * 4, 1);
}
