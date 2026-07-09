#include "pch.h"
#include "VxpgClusterPass.h"

#include "Constants.h"
#include "Utils/CVars.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgFingerprintPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

// Default 0 = SIByL-faithful frame-constant k-means++ seeding (its sampler is
// seeded with hardcoded zeros); 1 decorrelates the seeds per frame (ADR 0003).
static AutoCVarInt g_clusterFrameVaryingSeed("vxpg.cluster.frameVaryingSeed",
    "Re-randomize the k-means++ cluster seeds every frame (off = SIByL-faithful)",
    0, CVarFlags::EditCheckbox);

namespace
{
    constexpr uint32_t kClusterCount = 32;

    // PCG hash — matches Random.hlsl's pcg_hash (same construction as the
    // fingerprint pass's per-frame presample seed).
    uint32_t PcgHash(uint32_t state)
    {
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
}

void VxpgClusterPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgFingerprintPass>   fingerprintPass)
{
    spdlog::info("Initializing VXPG cluster pass...");

    m_device          = device;
    m_commandList     = commandList;
    m_voxelPass       = std::move(voxelPass);
    m_buildPass       = std::move(buildPass);
    m_fingerprintPass = std::move(fingerprintPass);

    CreateBuffers();
    CreateRootSignature();
    CreatePSOs();
    CreateCommandSignature();

    m_initialized = true;
}

void VxpgClusterPass::CreateBuffers()
{
    m_clusterSeedCompactIds = std::make_unique<RWStructuredBuffer<int32_t>>(
        m_device, kClusterCount, L"Cluster SeedCompactIds");
    m_clusterCenters = std::make_unique<RWStructuredBuffer<ClusterCenter>>(
        m_device, kClusterCount, L"Cluster Centers");
    m_voxelClusterAssignments = std::make_unique<RWStructuredBuffer<int32_t>>(
        m_device, Constants::Graphics::VOXEL_GUIDING_CAPACITY, L"Cluster VoxelClusterAssignments");
}

void VxpgClusterPass::CreateRootSignature()
{
    // Both kernels share one layout: b0 root constants (grid dim + seed frame
    // term), root UAVs u0 seeds, u1 centers, u2 dispatch args, u3 fingerprints,
    // u4 compact ids, u5 premul irradiance, u6 assignments.
    CD3DX12_ROOT_PARAMETER params[8];
    params[0].InitAsConstants(4, 0);
    for (uint32_t i = 0; i < 7; ++i)
        params[1 + i].InitAsUnorderedAccessView(i);

    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)));
    m_rootSig->SetName(L"VxpgCluster RootSig");
}

void VxpgClusterPass::CreatePSOs()
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

    createCsPso("resources/shaders/vxpgCluster.seed.shader",   L"VxpgCluster Seed PSO",   m_seedPso);
    createCsPso("resources/shaders/vxpgCluster.assign.shader", L"VxpgCluster Assign PSO", m_assignPso);
}

void VxpgClusterPass::CreateCommandSignature()
{
    D3D12_INDIRECT_ARGUMENT_DESC arg = {};
    arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC desc = {};
    desc.ByteStride       = sizeof(DirectX::XMUINT4); // one args entry (xyz + count)
    desc.NumArgumentDescs = 1;
    desc.pArgumentDescs   = &arg;

    ThrowIfFailed(m_device->CreateCommandSignature(&desc, nullptr,
        IID_PPV_ARGS(&m_dispatchCommandSignature)));
    m_dispatchCommandSignature->SetName(L"VxpgCluster DispatchCommandSignature");
}

void VxpgClusterPass::Run(uint32_t frameIndex)
{
    if (!m_initialized || !m_voxelPass || !m_buildPass || !m_fingerprintPass)
        return;

    auto* cmd = m_commandList.Get();

    const uint32_t seedFrameTerm =
        (g_clusterFrameVaryingSeed.Get() != 0) ? PcgHash(frameIndex) : 0u;
    uint32_t constants[4] = { m_voxelPass->GetGridDim(), seedFrameTerm, 0u, 0u };

    cmd->SetComputeRootSignature(m_rootSig.Get());
    cmd->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cmd->SetComputeRootUnorderedAccessView(1, m_clusterSeedCompactIds->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(2, m_clusterCenters->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(3, m_fingerprintPass->GetGuidingDispatchArgsBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(4, m_fingerprintPass->GetVoxelFingerprintsBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(5, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(6, m_buildPass->GetPremulIrradianceBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(7, m_voxelClusterAssignments->GetGPUVirtualAddress());

    // ---- Kernel 1: k-means++ seeding, one 1024-thread group ----
    cmd->SetPipelineState(m_seedPso.Get());
    cmd->Dispatch(1, 1, 1);
    m_clusterCenters->UavBarrier(cmd);
    m_clusterSeedCompactIds->UavBarrier(cmd);

    // ---- Kernel 2: nearest-center assignment for every compact voxel ----
    // Dispatched indirectly off gGuidingDispatchArgs[0] = (ceil(litVoxelCount/256),
    // 1, 1), replacing the worst-case ceil(CAPACITY/256)=512 fixed dispatch
    // (ADR 0003 option b). The fingerprint presample emitted the count this frame.
    cmd->SetPipelineState(m_assignPso.Get());

    // The args buffer (owned by the fingerprint pass) is currently a UAV; flip it
    // to INDIRECT_ARGUMENT for the ExecuteIndirect read, then back so the next
    // frame's presample writes it as a UAV again.
    ID3D12Resource* argsResource =
        m_fingerprintPass->GetGuidingDispatchArgsBuffer()->GetUnderlyingResource().Get();
    auto toIndirect = CD3DX12_RESOURCE_BARRIER::Transition(argsResource,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    cmd->ResourceBarrier(1, &toIndirect);

    cmd->ExecuteIndirect(m_dispatchCommandSignature.Get(), 1, argsResource,
        0, nullptr, 0); // entry [0]

    auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(argsResource,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &toUav);

    m_voxelClusterAssignments->UavBarrier(cmd);
}
