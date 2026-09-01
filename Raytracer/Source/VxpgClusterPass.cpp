#include "pch.h"
#include "Utils/GpuMemoryReport.h"
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
constexpr BindingSlot kClusterStats            = PassUav("gClusterStats", CLUSTER_REG_STATS);

constexpr BindingSlot kClusterSlots[] = {kClusterConstants,    kClusterSeedIds,      kClusterCenters,
                                         kClusterDispatchArgs, kClusterFingerprints, kClusterCompactIds,
                                         kClusterPremulIrradiance, kClusterAssignments, kClusterStats};
} // namespace

// Default 0 = SIByL-faithful frame-constant k-means++ seeding (its sampler is
// seeded with hardcoded zeros); 1 decorrelates the seeds per frame (ADR 0003).
static AutoCVarInt g_clusterFrameVaryingSeed("vxpg.cluster.frameVaryingSeed",
    "Re-randomize the k-means++ cluster seeds every frame (off = SIByL-faithful)",
    0, CVarFlags::EditCheckbox);

// One-shot: the armed frame counts cluster members and both distance terms, then
// logs and disarms. Off, the assign kernel branches the atomics out entirely, so
// a benchmark frame is unaffected.
static AutoCVarInt g_clusterDumpStats("vxpg.cluster.dumpStats",
    "Log the next frame's per-cluster population and Hamming/intensity split", 0, CVarFlags::EditCheckbox);

namespace
{
    constexpr uint32_t kClusterCount = 32;
    // population, Hamming, intensity (fixed point x1000), own-fingerprint popcount
    constexpr uint32_t kClusterStatFields = 4;
    constexpr uint32_t kClusterStatCount  = kClusterCount * kClusterStatFields;
    constexpr float    kClusterStatIntensityScale = 1000.0f;
    constexpr uint32_t kClusterStatMaxRetries = 120;

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
    m_clusterStats = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, kClusterStatCount, L"Cluster Stats");

    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    const auto bufferDesc     = CD3DX12_RESOURCE_DESC::Buffer(kClusterStatCount * sizeof(uint32_t));
    ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_clusterStatsReadback)));
    m_clusterStatsReadback->SetName(L"Cluster Stats Readback");
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
    uint32_t constants[4] = { m_voxelPass->GetGridDim(), seedFrameTerm,
                              IsStatsDumpArmed() ? 1u : 0u, 0u };

    cmd->SetComputeRootSignature(m_rootSig.Get());
    m_rootSig.SetConstants(cmd, kClusterConstants, constants, 4);
    m_rootSig.Set(cmd, kClusterSeedIds, m_clusterSeedCompactIds->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterCenters, m_clusterCenters->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterDispatchArgs, m_fingerprintPass->GetGuidingDispatchArgsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterFingerprints, m_fingerprintPass->GetVoxelFingerprintsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterCompactIds, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterPremulIrradiance, m_buildPass->GetPremulIrradianceBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterAssignments, m_voxelClusterAssignments->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kClusterStats, m_clusterStats->GetGPUVirtualAddress());

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

bool VxpgClusterPass::IsStatsDumpArmed()
{
    return g_clusterDumpStats.Get() != 0;
}

// Its own graph node, so the UNORDERED_ACCESS -> COPY_SOURCE flip is synthesized
// from the declaration rather than placed here.
void VxpgClusterPass::RecordStatsCopy()
{
    CommandContext::Get().GetCommandList()->CopyBufferRegion(
        m_clusterStatsReadback.Get(), 0,
        m_clusterStats->GetUnderlyingResource().Get(), 0,
        kClusterStatCount * sizeof(uint32_t));
}

void VxpgClusterPass::ResolveStats()
{
    if (!m_clusterStatsReadback)
        return;

    const D3D12_RANGE readRange{0, kClusterStatCount * sizeof(uint32_t)};
    uint32_t* stats = nullptr;
    if (FAILED(m_clusterStatsReadback->Map(0, &readRange, reinterpret_cast<void**>(&stats))))
        return;

    const uint32_t* population   = stats;
    const uint32_t* hammingSum   = stats + kClusterCount;
    const uint32_t* intensitySum = stats + kClusterCount * 2;
    const uint32_t* popcountSum  = stats + kClusterCount * 3;

    uint32_t total = 0;
    uint32_t occupied = 0;
    uint32_t largest = 0;
    double   totalHamming = 0.0;
    double   totalIntensity = 0.0;
    double   totalPopcount = 0.0;
    for (uint32_t cluster = 0; cluster < kClusterCount; ++cluster)
    {
        total += population[cluster];
        occupied += (population[cluster] > 0) ? 1u : 0u;
        largest = std::max(largest, population[cluster]);
        totalHamming += hammingSum[cluster];
        totalIntensity += intensitySum[cluster] / kClusterStatIntensityScale;
        totalPopcount += popcountSum[cluster];
    }

    const D3D12_RANGE writeRange{0, 0};

    // Armed from a command line, the first frame is frame 0 — before the bake and
    // the first injection, so there is nothing to count yet. Stay armed until a
    // frame has data rather than reporting an empty grid as a result; the cap
    // stops a genuinely unlit scene from flushing forever.
    if (total == 0 && m_statsRetries < kClusterStatMaxRetries)
    {
        ++m_statsRetries;
        m_clusterStatsReadback->Unmap(0, &writeRange);
        return;
    }

    spdlog::info("[VXPG cluster] {} lit voxels over {}/{} occupied clusters, largest holds {:.1f}%",
        total, occupied, kClusterCount, total > 0 ? 100.0 * largest / total : 0.0);

    // The bottom tree holds at most LIGHT_TREE_MAX_LEAVES compacted voxels
    // (vxpgLightTree.hlsl `min(rawCount, ...)`). Past that the guide simply cannot reach
    // the remainder, and nothing else says so: the shader raises an overflowFlag that no
    // CPU code reads. It bit at the old 32768 ceiling (128^3 on kitchen, 32986 lit voxels)
    // and again at 65536. The ceiling now equals VOXEL_GUIDING_CAPACITY, so compaction
    // drops the excess first and the second warning is the one that can still fire; both
    // stay, because they separate again the moment either constant moves.
    if (total > Constants::Graphics::LIGHT_TREE_MAX_LEAVES)
        spdlog::warn("[VXPG cluster] {} lit voxels exceeds the {}-leaf light tree by {} — the guide "
                     "cannot reach the excess, so this grid resolution measures the cap, not the grid",
                     total, Constants::Graphics::LIGHT_TREE_MAX_LEAVES,
                     total - Constants::Graphics::LIGHT_TREE_MAX_LEAVES);
    if (total > Constants::Graphics::VOXEL_GUIDING_CAPACITY)
        spdlog::warn("[VXPG cluster] {} lit voxels exceeds the {}-entry compaction buffer — voxels are "
                     "being dropped before clustering (voxelGuidingBuild.hlsl)",
                     total, Constants::Graphics::VOXEL_GUIDING_CAPACITY);

    if (total > 0)
    {
        // The metric weights the two terms equally, so whichever mean is larger is
        // the one actually choosing clusters. Fingerprint-driven grouping is the
        // whole point of the technique; an intensity-driven one is grouping by
        // brightness and would explain a spatially incoherent assignment.
        spdlog::info("[VXPG cluster] mean distance to assigned center: Hamming {:.2f} of 128 bits, intensity {:.2f}",
            totalHamming / total, totalIntensity / total);
        spdlog::info("[VXPG cluster] mean fingerprint popcount {:.1f} of 128 representatives visible per voxel",
            totalPopcount / total);
    }

    std::string histogram;
    for (uint32_t cluster = 0; cluster < kClusterCount; ++cluster)
        histogram += std::to_string(population[cluster]) + (cluster + 1 < kClusterCount ? " " : "");
    spdlog::info("[VXPG cluster] population: {}", histogram);

    m_clusterStatsReadback->Unmap(0, &writeRange);

    m_statsRetries = 0;
    g_clusterDumpStats.Set(0); // one-shot
}

// P5. The readback buffer is diagnostic and only exists while the cluster probe is armed,
// so it is reported when present rather than assumed away.
void VxpgClusterPass::ReportMemory(GpuMemoryReport& report) const
{
    using namespace GpuMemoryStage;
    report.Add(Cluster, "cluster seed compact ids",   m_clusterSeedCompactIds.get());
    report.Add(Cluster, "cluster centers",            m_clusterCenters.get());
    report.Add(Cluster, "voxel cluster assignments",  m_voxelClusterAssignments.get());
    report.Add(Cluster, "cluster stats",              m_clusterStats.get());
    report.Add(Cluster, "cluster stats readback",     m_clusterStatsReadback.Get());
}
