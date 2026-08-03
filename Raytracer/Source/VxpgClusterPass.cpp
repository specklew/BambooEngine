#include "pch.h"
#include "CommandContext.h"
#include "VxpgClusterPass.h"

#include "Constants.h"
#include "Utils/CVars.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgFingerprintPass.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
// Both cluster kernels share this layout: seeding writes the centers the
// assignment step then reads, so neither owns a subset.
constexpr BindingSlot kClusterConstants        = PassRootConstants("ClusterCB", CLUSTER_REG_CB, 4);
constexpr BindingSlot kClusterSeedIds          = PassUav("gClusterSeedCompactIds", CLUSTER_REG_SEED_COMPACT_IDS);
constexpr BindingSlot kClusterCenters          = PassUav("gClusterCenters", CLUSTER_REG_CENTERS);
constexpr BindingSlot kClusterDispatchArgs     = PassUav("gGuidingDispatchArgs", CLUSTER_REG_DISPATCH_ARGS);
constexpr BindingSlot kClusterFingerprints     = PassUav("gVoxelFingerprints", CLUSTER_REG_FINGERPRINTS);
constexpr BindingSlot kClusterCompactIds       = PassUav("gCompactIds", CLUSTER_REG_COMPACT_IDS);
constexpr BindingSlot kClusterPremulIrradiance = PassUav("gPremulIrradiance", CLUSTER_REG_PREMUL_IRRADIANCE);
constexpr BindingSlot kClusterAssignments      = PassUav("gVoxelClusterAssignments", CLUSTER_REG_ASSIGNMENTS);

constexpr BindingSlot kClusterSlots[] = {kClusterConstants,    kClusterSeedIds,      kClusterCenters,
                                         kClusterDispatchArgs, kClusterFingerprints, kClusterCompactIds,
                                         kClusterPremulIrradiance, kClusterAssignments};
} // namespace

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
    // Both kernels share one layout.
    m_rootSig = RootSignatureBuilder(L"VxpgCluster RootSig", /*tableCount*/ 0)
                    .Add(kClusterSlots)
                    .Build(m_device.Get());
}

void VxpgClusterPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_seedProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgCluster.seed.shader", L"VxpgCluster Seed PSO");
    m_assignProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgCluster.assign.shader", L"VxpgCluster Assign PSO");
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

bool VxpgClusterPass::BindCommon(uint32_t frameIndex)
{
    if (!m_initialized || !m_voxelPass || !m_buildPass || !m_fingerprintPass)
        return false;

    auto* cmd = m_commandList.Get();

    const uint32_t seedFrameTerm =
        (g_clusterFrameVaryingSeed.Get() != 0) ? PcgHash(frameIndex) : 0u;
    uint32_t constants[4] = { m_voxelPass->GetGridDim(), seedFrameTerm, 0u, 0u };

    cmd->SetComputeRootSignature(m_rootSig.Get());
    m_rootSig.SetConstants(cmd, kClusterConstants, constants, 4);
    m_rootSig.Set(cmd, kClusterSeedIds, m_clusterSeedCompactIds->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterCenters, m_clusterCenters->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterDispatchArgs, m_fingerprintPass->GetGuidingDispatchArgsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterFingerprints, m_fingerprintPass->GetVoxelFingerprintsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterCompactIds, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterPremulIrradiance, m_buildPass->GetPremulIrradianceBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterAssignments, m_voxelClusterAssignments->GetGPUVirtualAddress());

    return true;
}

// Kernel 1: k-means++ seeding, one 1024-thread group.
void VxpgClusterPass::RunSeed(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_seedProgram->GetPipelineState());
    CommandContext::Get().Dispatch(1, 1, 1);
}

// Kernel 2: nearest-center assignment for every compact voxel. Dispatched
// indirectly off gGuidingDispatchArgs[0] = (ceil(litVoxelCount/256), 1, 1),
// replacing the worst-case ceil(CAPACITY/256)=512 fixed dispatch (ADR 0003
// option b). The fingerprint presample emitted the count this frame; the args
// buffer's state flip is declared on this node rather than hand-placed here.
void VxpgClusterPass::RunAssign(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_assignProgram->GetPipelineState());
    CommandContext::Get().DispatchIndirect(m_dispatchCommandSignature.Get(),
        m_fingerprintPass->GetGuidingDispatchArgsBuffer()->GetUnderlyingResource().Get(), 0); // entry [0]
}
