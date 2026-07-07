#include "pch.h"
#include "VxpgClusterVisibilityPass.h"

#include "Constants.h"
#include "Renderer.h" // GetStaticSamplers
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgClusterPass.h"
#include "SuperpixelBuildPass.h"
#include "SceneResources/Scene.h"
#include "ResourceManager/ResourceManager.h"
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
    ComPtr<ID3D12DescriptorHeap>       globalHeap,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgClusterPass>       clusterPass,
    std::shared_ptr<SuperpixelBuildPass>   superpixelPass)
{
    spdlog::info("Initializing VXPG cluster-visibility pass...");

    m_device         = device;
    m_commandList    = commandList;
    m_globalHeap     = globalHeap;
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
    CD3DX12_DESCRIPTOR_RANGE r[10];
    r[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, 1);   // camera b0 @ heap 1
    r[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 3);   // TLAS t0 @ 3
    r[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, 4);   // vertices t1 @ 4
    r[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, 5);   // indices t2 @ 5
    r[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3, 0, Constants::Graphics::SUPERPIXEL_INDEX_DESCRIPTOR_INDEX);        // gSuperpixelIndex u3 @ 523
    r[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, Constants::Graphics::VPL_POSITION_DESCRIPTOR_INDEX);           // gVplPosition u1 @ 526
    r[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2, 0, Constants::Graphics::VBUFFER_DESCRIPTOR_INDEX);                // gVBuffer u2 @ 527
    r[7].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 4, 0, Constants::Graphics::SPIXEL_GATHERED_DESCRIPTOR_INDEX);        // gSpixelGathered u4 @ 528
    r[8].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 5, 0, Constants::Graphics::SPIXEL_COUNTER_DESCRIPTOR_INDEX);         // gSpixelCounter u5 @ 529
    r[9].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 6, 0, Constants::Graphics::CLUSTER_VISIBILITY_MASK_DESCRIPTOR_INDEX); // gClusterVisibilityMask u6 @ 530

    CD3DX12_ROOT_PARAMETER params[10];
    params[0].InitAsDescriptorTable(_countof(r), r);
    params[1].InitAsShaderResourceView(3);       // geometry info t3
    params[2].InitAsShaderResourceView(4);       // instance info t4
    params[3].InitAsConstantBufferView(1);       // grid CB b1
    params[4].InitAsConstants(8, 2);             // cvis constants b2
    params[5].InitAsUnorderedAccessView(7);      // gVoxInverseIndex u7
    params[6].InitAsUnorderedAccessView(8);      // gVoxelClusterAssignments u8
    params[7].InitAsUnorderedAccessView(9);      // gClusterGatheredLightPoints u9
    params[8].InitAsUnorderedAccessView(10);     // gClusterLightPointCounts u10
    params[9].InitAsUnorderedAccessView(11);     // gSpixelClusterAvgVisibility u11

    auto samplers = Renderer::GetStaticSamplers();
    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params,
        static_cast<UINT>(samplers.size()), samplers.data(),
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)));
    m_rootSig->SetName(L"VxpgClusterVisibility RootSig");
}

void VxpgClusterVisibilityPass::CreatePSOs()
{
    auto& rm = ResourceManager::Get();

    auto createCsPso = [&](const char* assetPath, const wchar_t* name,
                           ComPtr<ID3D12PipelineState>& out)
    {
        auto handle = rm.GetOrLoadShader(AssetId(assetPath));
        auto blob = rm.shaders.GetResource(handle).bytecode;

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_rootSig.Get();
        desc.CS = CD3DX12_SHADER_BYTECODE(blob->GetBufferPointer(), blob->GetBufferSize());
        ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out)));
        out->SetName(name);
    };

    createCsPso("resources/shaders/vxpgClusterVisibility.clear.shader",  L"Cvis Clear PSO",  m_clearPso);
    createCsPso("resources/shaders/vxpgClusterVisibility.gather.shader", L"Cvis Gather PSO", m_gatherPso);
    createCsPso("resources/shaders/vxpgClusterVisibility.check.shader",  L"Cvis Check PSO",  m_checkPso);
}

void VxpgClusterVisibilityPass::OnResize(uint32_t width, uint32_t height)
{
    m_width  = width;
    m_height = height;
    m_mapX   = (width  + kSuperpixelSize - 1) / kSuperpixelSize;
    m_mapY   = (height + kSuperpixelSize - 1) / kSuperpixelSize;
    CreateResolutionBuffers();
}

void VxpgClusterVisibilityPass::Run(uint32_t frameIndex)
{
    if (!m_initialized || !m_scene || !m_voxelPass || !m_buildPass ||
        !m_clusterPass || !m_superpixelPass || m_mapX == 0)
        return;

    auto* cmd = m_commandList.Get();

    ID3D12DescriptorHeap* heaps[] = { m_globalHeap.Get() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    cmd->SetComputeRootSignature(m_rootSig.Get());
    cmd->SetComputeRootDescriptorTable(0, m_globalHeap->GetGPUDescriptorHandleForHeapStart());
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

    auto maskBarrier = [&]() {
        auto b = CD3DX12_RESOURCE_BARRIER::UAV(m_mask.Get());
        cmd->ResourceBarrier(1, &b);
    };

    // ---- Clear ----
    cmd->SetPipelineState(m_clearPso.Get());
    cmd->Dispatch((m_mapX + 15) / 16, (m_mapY + 15) / 16, 1);
    maskBarrier();
    m_clusterLightPointCounts->UavBarrier(cmd);
    m_avgVisibility->UavBarrier(cmd);

    // ---- Gather: file VPLs into cluster drawers + seed mask bits ----
    cmd->SetPipelineState(m_gatherPso.Get());
    cmd->Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);
    maskBarrier();
    m_clusterGatheredLightPoints->UavBarrier(cmd);
    m_clusterLightPointCounts->UavBarrier(cmd);

    // ---- Check: shadow-ray probes -> soft avg-visibility + mask ----
    // Dispatch covers mapX superpixels wide (32 sample lanes each) and mapY*4
    // groups tall (8 clusters per group x 4 = 32 clusters per superpixel row).
    cmd->SetPipelineState(m_checkPso.Get());
    cmd->Dispatch(m_mapX, m_mapY * 4, 1);
    maskBarrier();
    m_avgVisibility->UavBarrier(cmd);
}
