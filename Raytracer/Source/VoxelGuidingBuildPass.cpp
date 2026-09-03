#include "pch.h"
#include "Utils/GpuMemoryReport.h"
#include "CommandContext.h"
#include "VoxelGuidingBuildPass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
// The three voxel textures are read out of the global heap, at the same slots
// their producers (voxelization, light injection) write; everything else is a
// root descriptor. All three kernels share the layout.
constexpr BindingSlot kGuidingBuildConstants = PassRootConstants("BuildCB", GUIDING_BUILD_REG_CB, 4);
constexpr BindingSlot kGuidingBuildIrradiance =
    PassTableEntry("gVoxIrradiance", BindingKind::Uav, GUIDING_BUILD_REG_IRRADIANCE, GlobalDescriptor::VoxelIrradiance);
constexpr BindingSlot kGuidingBuildVplCount =
    PassTableEntry("gVoxVplCount", BindingKind::Uav, GUIDING_BUILD_REG_VPL_COUNT, GlobalDescriptor::VoxelVplCount);
constexpr BindingSlot kGuidingBuildRepresentative =
    PassTableEntry("gVoxelRepresentative", BindingKind::Uav, GUIDING_BUILD_REG_VOXEL_REPRESENTATIVE,
                   GlobalDescriptor::VoxelRepresentative);
constexpr BindingSlot kGuidingBuildCounters      = PassUav("gCounters", GUIDING_BUILD_REG_COUNTERS);
// Read only by the census (see the probe): the denominator of the lit-voxel share is the
// number of cells the bake marked as holding geometry, and nothing else counted them.
constexpr BindingSlot kGuidingBuildOccupancy =
    PassTableEntry("gOccupancy", BindingKind::Uav, GUIDING_BUILD_REG_OCCUPANCY, GlobalDescriptor::VoxelOccupancy);
// [0..3] the compaction count and the accumulator probe; [4..6] the unsteerable-share
// probe (second moment where the guide has no density, its total, and how many samples hit
// the fixed-point ceiling).
// [0..3] compaction + accumulator probe; [16..79] and [80..143] the two halves of the
// unsteerable probe's log2 histogram.
constexpr uint32_t kCounterCount = 144;
constexpr uint32_t kUnsteerableBuckets = 64;
constexpr uint32_t kUnsteerableTotalBase = 16;
constexpr uint32_t kUnsteerableBlockedBase = 80;
// Retried because the first frame runs before the bake and the first injection, so an
// unarmed-looking zero is indistinguishable from a genuinely unlit grid on frame 0.
constexpr uint32_t kProbeMaxRetries = 8;
}

// One-shot: the compaction kernel already reads both voxel textures, so arming this costs
// two atomics on cells it was visiting anyway and nothing at all when disarmed.
// The S1 ceiling of the guided estimator. On directions that hit no lit voxel the guide has
// zero density, the balance weight collapses to 1 and the estimator IS the BSDF estimator, so
// the share of the BSDF strategy's SECOND MOMENT that lands there bounds what any guide can
// win: Var_B / Var_MIS <= 1 / sigma_u. A scene whose sigma_u exceeds the reciprocal of the
// frame-cost multiplier cannot win at equal time however well the guide is built - which turns
// "render both arms and see" into a prediction. Second moment and not energy: a uniform sky
// carries much of the energy and almost none of the variance.
static AutoCVarInt g_unsteerableProbe("vxpg.guiding.unsteerable",
    "One-shot readout of the share of the BSDF strategy's second moment that the guide cannot reach",
    0, CVarFlags::EditCheckbox);

static AutoCVarInt g_guidingProbe("vxpg.guiding.probe",
    "One-shot readout of the irradiance accumulator: headroom to a uint32 wrap and the share of "
    "VPL-carrying cells that quantise to zero", 0, CVarFlags::EditCheckbox);

namespace
{
constexpr BindingSlot kGuidingBuildCompactIds    = PassUav("gCompactIds", GUIDING_BUILD_REG_COMPACT_IDS);
constexpr BindingSlot kGuidingBuildInverseIndex  = PassUav("gInverseIndex", GUIDING_BUILD_REG_INVERSE_INDEX);
constexpr BindingSlot kGuidingBuildLiveBoundMin  = PassUav("gLiveBoundMin", GUIDING_BUILD_REG_LIVE_BOUND_MIN);
constexpr BindingSlot kGuidingBuildLiveBoundMax  = PassUav("gLiveBoundMax", GUIDING_BUILD_REG_LIVE_BOUND_MAX);
constexpr BindingSlot kGuidingBuildLightPoints =
    PassUav("gCompactVoxelLightPoints", GUIDING_BUILD_REG_COMPACT_LIGHT_POINTS);
constexpr BindingSlot kGuidingBuildPremulIrradiance = PassUav("gPremulIrradiance", GUIDING_BUILD_REG_PREMUL_IRRADIANCE);
constexpr BindingSlot kGuidingBuildBakedBoundMin    = PassUav("gBakedBoundMin", GUIDING_BUILD_REG_BAKED_BOUND_MIN);
constexpr BindingSlot kGuidingBuildBakedBoundMax    = PassUav("gBakedBoundMax", GUIDING_BUILD_REG_BAKED_BOUND_MAX);

constexpr BindingSlot kGuidingBuildSlots[] = {
    kGuidingBuildConstants,     kGuidingBuildIrradiance,   kGuidingBuildVplCount,      kGuidingBuildRepresentative,
    kGuidingBuildCounters,      kGuidingBuildCompactIds,   kGuidingBuildInverseIndex,  kGuidingBuildLiveBoundMin,
    kGuidingBuildLiveBoundMax,  kGuidingBuildLightPoints,  kGuidingBuildPremulIrradiance,
    kGuidingBuildBakedBoundMin, kGuidingBuildBakedBoundMax, kGuidingBuildOccupancy};
} // namespace

void VoxelGuidingBuildPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>  voxelPass)
{
    spdlog::info("Initializing voxel guiding build pass...");

    m_device      = device;
    m_commandList = commandList;
    m_voxelPass   = std::move(voxelPass);

    CreateBuffers();
    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void VoxelGuidingBuildPass::CreateBuffers()
{
    constexpr uint32_t capacity = Constants::Graphics::VOXEL_GUIDING_CAPACITY;

    // [0] compacted voxel count, [1] retired (was the CDF total weight), [2] and [3] the
    // one-shot accumulator probe. Downstream root-UAV consumers only ever index [0].
    m_counters     = std::make_unique<RWStructuredBuffer<uint32_t>>(m_device, kCounterCount, L"VoxelGuiding Counters");

    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    const auto bufferDesc     = CD3DX12_RESOURCE_DESC::Buffer(kCounterCount * sizeof(uint32_t));
    ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_countersReadback)));
    m_countersReadback->SetName(L"VoxelGuiding Counters Readback");
    m_compactIds   = std::make_unique<RWStructuredBuffer<uint32_t>>(m_device, capacity, L"VoxelGuiding CompactIds");
    m_compactVoxelLightPoints = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT4>>(m_device, capacity, L"VoxelGuiding CompactVoxelLightPoints");
    m_premulIrradiance = std::make_unique<RWStructuredBuffer<float>>(m_device, capacity, L"VoxelGuiding PremulIrradiance");

    const auto premulDesc = CD3DX12_RESOURCE_DESC::Buffer(capacity * sizeof(float));
    ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &premulDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_premulReadback)));
    m_premulReadback->SetName(L"VoxelGuiding PremulIrradiance Readback");

    CreateGridSizedBuffers();
}

void VoxelGuidingBuildPass::CreateGridSizedBuffers()
{
    // Grid-sized (one element per cell), not capacity-sized. Bound as ROOT
    // UAVs (no bounds checking), so they MUST track the grid dim exactly —
    // undersized means the shaders write past the end and corrupt GPU memory.
    const uint32_t gridDim = m_voxelPass->GetGridDim();
    const uint32_t cellCount = gridDim * gridDim * gridDim;
    m_inverseIndex = std::make_unique<RWStructuredBuffer<int32_t>>(m_device, cellCount, L"VoxelGuiding InverseIndex");
    m_liveBoundMin = std::make_unique<RWStructuredBuffer<DirectX::XMUINT4>>(m_device, cellCount, L"VoxelGuiding LiveBoundMin");
    m_liveBoundMax = std::make_unique<RWStructuredBuffer<DirectX::XMUINT4>>(m_device, cellCount, L"VoxelGuiding LiveBoundMax");
}

void VoxelGuidingBuildPass::OnVoxelGridResize()
{
    if (!m_initialized)
        return;
    CreateGridSizedBuffers();
}

void VoxelGuidingBuildPass::CreateRootSignature()
{
    m_rootSig = RootSignatureBuilder(L"VoxelGuidingBuild RootSig", /*tableCount*/ 1)
                    .Add(kGuidingBuildSlots)
                    .Build(m_device.Get());
}

void VoxelGuidingBuildPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/voxelGuidingBuild.clear.shader", L"VoxelGuiding Clear PSO");
    m_reloadProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/voxelGuidingBuild.reload.shader", L"VoxelGuiding Reload PSO");
    m_compactProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/voxelGuidingBuild.compact.shader", L"VoxelGuiding Compact PSO");
}

bool VoxelGuidingBuildPass::BindCommon()
{
    if (!m_initialized || !m_voxelPass)
        return false;

    const uint32_t gridDim = m_voxelPass->GetGridDim();

    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSig.Get());

    // The census walks the whole grid, so it rides the probe's arming rather than
    // running every frame: disarmed, the compaction kernel skips the occupancy read.
    uint32_t constants[4] = { gridDim, IsProbeArmed() ? 1u : 0u, 0, 0 };
    auto* cmd = m_commandList.Get();
    m_rootSig.SetConstants(cmd, kGuidingBuildConstants, constants, 4);
    m_rootSig.SetTable(cmd, 0, globalHeap.GpuStart());
    m_rootSig.Set(cmd, kGuidingBuildCounters, m_counters->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildCompactIds, m_compactIds->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildInverseIndex, m_inverseIndex->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildLiveBoundMin, m_liveBoundMin->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildLiveBoundMax, m_liveBoundMax->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildLightPoints, m_compactVoxelLightPoints->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildPremulIrradiance, m_premulIrradiance->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildBakedBoundMin, m_voxelPass->GetBakedBoundMinBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildBakedBoundMax, m_voxelPass->GetBakedBoundMaxBuffer()->GetGPUVirtualAddress());

    return true;
}

void VoxelGuidingBuildPass::RunClear()
{
    if (!BindCommon())
        return;

    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());
    CommandContext::Get().Dispatch(1, 1, 1);
}

// Reload baked bounds for lit voxels before compaction reads them.
void VoxelGuidingBuildPass::RunReload()
{
    if (!BindCommon())
        return;

    const uint32_t groups = (m_voxelPass->GetGridDim() + 7) / 8;
    m_commandList->SetPipelineState(m_reloadProgram->GetPipelineState());
    CommandContext::Get().Dispatch(groups, groups, groups);
}

void VoxelGuidingBuildPass::RunCompact()
{
    if (!BindCommon())
        return;

    const uint32_t groups = (m_voxelPass->GetGridDim() + 7) / 8;
    m_commandList->SetPipelineState(m_compactProgram->GetPipelineState());
    CommandContext::Get().Dispatch(groups, groups, groups);
}

// P5. Two of these are grid-sized (the inverse index and both live bounds) and the rest
// are capped by VOXEL_GUIDING_CAPACITY, so the stage has a fixed part and a cubic part.
bool VoxelGuidingBuildPass::IsUnsteerableArmed()
{
    return g_unsteerableProbe.Get() != 0;
}

bool VoxelGuidingBuildPass::IsProbeArmed()
{
    return g_guidingProbe.Get() != 0 || g_unsteerableProbe.Get() != 0;
}

// Its own graph node, so the UNORDERED_ACCESS -> COPY_SOURCE flip is synthesized from the
// declaration rather than placed here (same shape as the cluster probe).
void VoxelGuidingBuildPass::RecordProbeCopy()
{
    CommandContext::Get().GetCommandList()->CopyBufferRegion(
        m_countersReadback.Get(), 0,
        m_counters->GetUnderlyingResource().Get(), 0,
        kCounterCount * sizeof(uint32_t));

    // The premultiplied irradiance of every compacted voxel - the quantity the light tree
    // weights its choice by, so the right one to ask how concentrated the light is.
    if (m_premulReadback && m_premulIrradiance)
        CommandContext::Get().GetCommandList()->CopyBufferRegion(
            m_premulReadback.Get(), 0,
            m_premulIrradiance->GetUnderlyingResource().Get(), 0,
            Constants::Graphics::VOXEL_GUIDING_CAPACITY * sizeof(float));
}

void VoxelGuidingBuildPass::LogConcentration(uint32_t litVoxels) const
{
    if (!m_premulReadback || litVoxels == 0)
        return;

    const uint32_t count = std::min(litVoxels,
        static_cast<uint32_t>(Constants::Graphics::VOXEL_GUIDING_CAPACITY));
    const D3D12_RANGE readRange{0, count * sizeof(float)};
    float* mapped = nullptr;
    if (FAILED(m_premulReadback->Map(0, &readRange, reinterpret_cast<void**>(&mapped))))
        return;

    std::vector<double> weights;
    weights.reserve(count);
    double total = 0.0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const double value = static_cast<double>(mapped[i]);
        if (value > 0.0)
        {
            weights.push_back(value);
            total += value;
        }
    }
    const D3D12_RANGE writeRange{0, 0};
    m_premulReadback->Unmap(0, &writeRange);

    if (weights.empty() || total <= 0.0)
        return;

    std::sort(weights.begin(), weights.end(), std::greater<double>());

    // At least one voxel per bucket: on a scene with 49 lit voxels, 1 % rounds to zero and
    // the answer would read as "0 % of the energy" instead of "whatever the brightest holds".
    auto headShare = [&](int percent)
    {
        const size_t take = std::max<size_t>(1, (weights.size() * percent) / 100);
        double head = 0.0;
        for (size_t i = 0; i < take; ++i)
            head += weights[i];
        return 100.0 * head / total;
    };

    // Participation ratio: the number of voxels the distribution behaves as if it spread over
    // evenly. Equals the count for a flat distribution and 1 for a single spike.
    double sumOfSquares = 0.0;
    for (const double weight : weights)
        sumOfSquares += (weight / total) * (weight / total);
    const double effective = 1.0 / sumOfSquares;

    spdlog::info("[VXPG concentration] top 1% holds {:.1f}%, top 5% {:.1f}%, top 10% {:.1f}% "
                 "of total irradiance; effective support {:.0f} of {} lit voxels ({:.2f}%)",
                 headShare(1), headShare(5), headShare(10), effective, weights.size(),
                 100.0 * effective / static_cast<double>(weights.size()));
}

void VoxelGuidingBuildPass::ResolveProbe()
{
    if (!m_countersReadback)
        return;

    const D3D12_RANGE readRange{0, kCounterCount * sizeof(uint32_t)};
    uint32_t* counters = nullptr;
    if (FAILED(m_countersReadback->Map(0, &readRange, reinterpret_cast<void**>(&counters))))
        return;

    const uint32_t litVoxels    = counters[0];
    const uint32_t occupied     = counters[1];
    const uint32_t truncated    = counters[2];
    const uint32_t maxPackedSum = counters[3];

    const D3D12_RANGE writeRange{0, 0};
    if (litVoxels == 0 && truncated == 0 && m_probeRetries < kProbeMaxRetries)
    {
        ++m_probeRetries;
        m_countersReadback->Unmap(0, &writeRange);
        return;
    }

    // The accumulator is a uint32 that InterlockedAdd never saturates, so the question is
    // how much headroom the brightest cell actually leaves. Reported as a share of the
    // wrap point rather than a raw number, because the raw number means nothing alone.
    const double headroom = 100.0 * static_cast<double>(maxPackedSum) / 4294967295.0;
    spdlog::info("[VXPG accumulator] brightest cell holds {} of 4294967295 ({:.4f}% of the wrap point)",
                 maxPackedSum, headroom);
    if (headroom > 10.0)
        spdlog::warn("[VXPG accumulator] within one order of magnitude of wrapping — a wrapped cell "
                     "reads as nearly unlit in the brightest part of the scene");

    // PackIrradiance truncates, so a cell whose samples all fall below 1/100 packs to zero,
    // and the compaction below drops it. Harmless while the share is small (those cells carry
    // almost no energy); a large share means the guide is losing support to quantisation.
    const uint32_t caughtVpls = litVoxels + truncated;
    spdlog::info("[VXPG accumulator] {} of {} cells with VPLs packed to zero and left the guide ({:.2f}%)",
                 truncated, caughtVpls, caughtVpls > 0 ? 100.0 * truncated / caughtVpls : 0.0);

    // The scene-selection metric of the evaluation plan: lit cells over cells holding
    // geometry. Both counts are of the SAME grid, so the share is comparable across
    // resolutions in a way neither count is on its own.
    spdlog::info("[VXPG census] {} lit voxels of {} occupied by geometry ({:.2f}% lit)",
                 litVoxels, occupied, occupied > 0 ? 100.0 * litVoxels / occupied : 0.0);

    LogConcentration(litVoxels);

    // Past the capacity the compaction keeps the first VOXEL_GUIDING_CAPACITY voxels and
    // drops the rest, and every consumer clamps to that ceiling — so the cluster pass now
    // reports exactly the ceiling and cannot see the overflow. The raw count only exists
    // here, which makes this the only place the truncation can still be reported.
    if (litVoxels > Constants::Graphics::VOXEL_GUIDING_CAPACITY)
        spdlog::warn("[VXPG census] {} lit voxels exceeds the {}-entry compaction buffer by {} — this "
                     "grid resolution measures the cap, not the grid",
                     litVoxels, Constants::Graphics::VOXEL_GUIDING_CAPACITY,
                     litVoxels - Constants::Graphics::VOXEL_GUIDING_CAPACITY);

    m_countersReadback->Unmap(0, &writeRange);
    m_probeRetries = 0;
    g_guidingProbe.Set(0); // one-shot
}

// Read after the integrator, not after the build chain: the accumulators are filled by the
// raygen, so the copy node for this one sits past the technique (see Renderer).
void VoxelGuidingBuildPass::ResolveUnsteerable()
{
    if (!m_countersReadback)
        return;

    const D3D12_RANGE readRange{0, kCounterCount * sizeof(uint32_t)};
    uint32_t* counters = nullptr;
    if (FAILED(m_countersReadback->Map(0, &readRange, reinterpret_cast<void**>(&counters))))
        return;

    // Rebuild both sums from the histogram, weighting each octave by its geometric
    // midpoint. Sharing the buckets between numerator and denominator is what makes the
    // ratio insensitive to the octave width.
    double unreachable = 0.0;
    double total = 0.0;
    double fireflyMass = 0.0;
    uint64_t samples = 0;
    for (uint32_t bucket = 0; bucket < kUnsteerableBuckets; ++bucket)
    {
        const double midpoint = std::pow(2.0, static_cast<double>(bucket) - 32.0 + 0.5);
        const double all     = counters[kUnsteerableTotalBase + bucket] * midpoint;
        const double blocked = counters[kUnsteerableBlockedBase + bucket] * midpoint;
        total += all;
        unreachable += blocked;
        samples += counters[kUnsteerableTotalBase + bucket];
        // "Firefly" here means a squared contribution above 1, i.e. a sample brighter than
        // the whole rest of the image's typical scale.
        if (bucket >= 32)
            fireflyMass += all;
    }
    const double largeShare = total > 0.0 ? fireflyMass / total : 0.0;
    const uint32_t clipped = 0;

    const D3D12_RANGE writeRange{0, 0};
    if (total <= 0.0 && m_unsteerableRetries < kProbeMaxRetries)
    {
        ++m_unsteerableRetries;
        m_countersReadback->Unmap(0, &writeRange);
        return;
    }
    m_countersReadback->Unmap(0, &writeRange);

    if (total <= 0.0)
    {
        spdlog::warn("[VXPG unsteerable] no BSDF-strategy samples accumulated - is the guided "
                     "technique active and the debug view off?");
        g_unsteerableProbe.Set(0);
        m_unsteerableRetries = 0;
        return;
    }

    const double share = unreachable / total;
    if (share > 0.0)
        spdlog::info("[VXPG unsteerable] sigma_u = {:.5f} over {} samples; ceiling on "
                     "Var_B/Var_MIS = {:.1f}x; {:.1f}% of the second moment above squared 1",
                     share, samples, 1.0 / share, 100.0 * largeShare);
    else
        spdlog::info("[VXPG unsteerable] sigma_u = 0 over {} samples: every sample carrying "
                     "second moment landed where the guide has density, so this ceiling does "
                     "not bind at all; {:.1f}% of the second moment above squared 1",
                     samples, 100.0 * largeShare);
    (void)clipped;

    g_unsteerableProbe.Set(0); // one-shot
    m_unsteerableRetries = 0;
}

void VoxelGuidingBuildPass::ReportMemory(GpuMemoryReport& report) const
{
    using namespace GpuMemoryStage;
    report.Add(GuidingBuild, "counters",              m_counters.get());
    report.Add(GuidingBuild, "compact ids",           m_compactIds.get());
    report.Add(GuidingBuild, "inverse index",         m_inverseIndex.get());
    report.Add(GuidingBuild, "live bound min",        m_liveBoundMin.get());
    report.Add(GuidingBuild, "live bound max",        m_liveBoundMax.get());
    report.Add(GuidingBuild, "compact light points",  m_compactVoxelLightPoints.get());
    report.Add(GuidingBuild, "premultiplied irradiance", m_premulIrradiance.get());
}
