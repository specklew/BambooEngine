#include "pch.h"
#include "CommandContext.h"
#include "Techniques/GuidedPathTracingPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "FrameBindingLayout.h"
#include "GlobalDescriptorHeap.h"
#include "GuidingDebugView.h"
#include "PassRegisters.h"
#include "Renderer.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/CVars.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgFingerprintPass.h"
#include "VxpgClusterPass.h"
#include "VxpgLightTreePass.h"
#include "Window.h"
#include "Resources/ShaderBindingTable.h"
#include "Resources/StructuredBuffer.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"

// -1 = auto (inline RayQuery on AMD RDNA, RT pipeline elsewhere — RDNA's
// shader-based traversal skips the pipeline/SBT machinery nearly for free,
// while NVIDIA schedules the RT pipeline well and SER, when Ada+ hardware is
// present, only exists in the pipeline model). 0/1 force. ADR 0011.
static AutoCVarInt g_inlineRayQuery("vxpg.integrator.inlineRq",
    "Guided integrator backend: -1 auto by GPU vendor, 0 RT pipeline, 1 inline RayQuery compute",
    -1, CVarFlags::EditDrag);

namespace
{
// Pass bindings, on top of the frame layout. One list: the signature is built
// from it and every bind names a slot out of it, so there is no parallel index
// sequence to keep in step.
//
// Texture UAVs cannot be root descriptors, so they ride the shared heap table at
// their global slots; everything else is a root descriptor.
//
// Each slot also carries what the integrator does to it, which is what the frame
// graph declares (ADR 0017 step 3). Four of them are WRITES: under injection
// reuse (ADR 0009) this raygen's BSDF subtree owns the VPL data, so the
// integrator is their producer and light injection returns before writing any.
constexpr GraphAccess kRead  = GraphAccess::UnorderedAccessRead;
constexpr GraphAccess kWrite = GraphAccess::ComputeWrite;

constexpr BindingSlot kVoxelIrradiance = Accesses(PassTableEntry("gVoxIrradiance", BindingKind::Uav,
    GUIDED_REG_IRRADIANCE, GlobalDescriptor::VoxelIrradiance), kWrite);
constexpr BindingSlot kVoxelVplCount = Accesses(PassTableEntry("gVoxVplCount", BindingKind::Uav,
    GUIDED_REG_VPL_COUNT, GlobalDescriptor::VoxelVplCount), kWrite);
constexpr BindingSlot kVoxelRepresentative = Accesses(PassTableEntry("gVoxelRepresentative", BindingKind::Uav,
    GUIDED_REG_VOXEL_REPRESENTATIVE, GlobalDescriptor::VoxelRepresentative), kWrite);
constexpr BindingSlot kVplPosition = Accesses(PassTableEntry("gVplPosition", BindingKind::Uav,
    GUIDED_REG_VPL_POSITION, GlobalDescriptor::VplPosition), kWrite);

constexpr BindingSlot kSuperpixelIndex = Accesses(PassTableEntry("gSpixelIndexImage", BindingKind::Uav,
    GUIDED_REG_SUPERPIXEL_INDEX, GlobalDescriptor::SuperpixelIndex), kRead);
constexpr BindingSlot kVBuffer = Accesses(PassTableEntry("gVBuffer", BindingKind::Uav,
    GUIDED_REG_VBUFFER, GlobalDescriptor::VBuffer), kRead);
constexpr BindingSlot kVisibilityMask = Accesses(PassTableEntry("gClusterVisibilityMask", BindingKind::Uav,
    GUIDED_REG_VISIBILITY_MASK, GlobalDescriptor::ClusterVisibilityMask), kRead);
constexpr BindingSlot kFuzzyWeights = Accesses(PassTableEntry("gFuzzyWeights", BindingKind::Uav,
    GUIDED_REG_FUZZY_WEIGHT, GlobalDescriptor::FuzzyWeight), kRead);
constexpr BindingSlot kFuzzyIndices = Accesses(PassTableEntry("gFuzzyIndices", BindingKind::Uav,
    GUIDED_REG_FUZZY_INDEX, GlobalDescriptor::FuzzyIndex), kRead);
constexpr BindingSlot kGuidingCounters     = Accesses(PassUav("gVoxCounters", GUIDED_REG_COUNTERS), kRead);
constexpr BindingSlot kGuidingCompactIds   = Accesses(PassUav("gVoxCompactIds", GUIDED_REG_COMPACT_IDS), kRead);
constexpr BindingSlot kGuidingInverseIndex = Accesses(PassUav("gVoxInverseIndex", GUIDED_REG_INVERSE_INDEX), kRead);
constexpr BindingSlot kVoxelFingerprints   = Accesses(PassUav("gVoxelFingerprints", GUIDED_REG_FINGERPRINTS), kRead);
constexpr BindingSlot kClusterAssignments  = Accesses(PassUav("gVoxelClusterAssignments", GUIDED_REG_CLUSTER_ASSIGNMENTS), kRead);
constexpr BindingSlot kClusterSeeds        = Accesses(PassUav("gClusterSeedCompactIds", GUIDED_REG_CLUSTER_SEEDS), kRead);
constexpr BindingSlot kLightTreeNodes      = Accesses(PassUav("gLightTreeNodes", GUIDED_REG_LIGHT_TREE_NODES), kRead);
constexpr BindingSlot kCompactToLeaf       = Accesses(PassUav("gCompactToLeaf", GUIDED_REG_COMPACT_TO_LEAF), kRead);
constexpr BindingSlot kClusterRoots        = Accesses(PassUav("gClusterRootNodes", GUIDED_REG_CLUSTER_ROOTS), kRead);
constexpr BindingSlot kImportanceHeap      = Accesses(PassUav("gSpixelClusterImportanceHeap", GUIDED_REG_IMPORTANCE_HEAP), kRead);
constexpr BindingSlot kLiveBoundMin        = Accesses(PassUav("gVoxelLiveBoundMin", GUIDED_REG_LIVE_BOUND_MIN), kRead);
constexpr BindingSlot kLiveBoundMax        = Accesses(PassUav("gVoxelLiveBoundMax", GUIDED_REG_LIVE_BOUND_MAX), kRead);

// No graph access: constants have no producer, and the adaptive-q pair is read by
// the dispatch but written by the update node, which one slot cannot say.
constexpr BindingSlot kVoxelGridConstants = PassCbv("VoxelGridCB", REG_VOXEL_GRID_CB);
constexpr BindingSlot kTileGuideQ         = PassUav("gTileGuideQ", GUIDED_REG_TILE_GUIDE_Q);         // ADR 0015
constexpr BindingSlot kTileStrategyStats  = PassUav("gTileStrategyStats", GUIDED_REG_TILE_STRATEGY_STATS);
constexpr BindingSlot kAdaptiveQConstants = PassRootConstants("AdaptiveQCB", GUIDED_REG_ADAPTIVE_Q_CB, 1);

constexpr BindingSlot kGuidedSlots[] = {
    kVoxelIrradiance,     kVoxelVplCount,   kSuperpixelIndex,  kVoxelRepresentative, kVplPosition,
    kVBuffer,             kVisibilityMask,  kFuzzyWeights,     kFuzzyIndices,        kVoxelGridConstants,
    kGuidingCounters,     kGuidingCompactIds, kGuidingInverseIndex, kVoxelFingerprints, kClusterAssignments,
    kClusterSeeds,        kLightTreeNodes,  kCompactToLeaf,    kClusterRoots,        kImportanceHeap,
    kLiveBoundMin,        kLiveBoundMax,    kTileGuideQ,       kTileStrategyStats,   kAdaptiveQConstants};

// The graph-visible subset, paired with the frame's handles below. Everything the
// signature binds appears here exactly once — that is the property that made the
// two hand-written lists drift apart.
constexpr BindingSlot kGuidedGraphSlots[] = {
    kVoxelIrradiance,   kVoxelVplCount,     kVoxelRepresentative, kVplPosition,     kSuperpixelIndex,
    kVBuffer,           kVisibilityMask,    kFuzzyWeights,        kFuzzyIndices,    kGuidingCounters,
    kGuidingCompactIds, kGuidingInverseIndex, kVoxelFingerprints, kClusterAssignments, kClusterSeeds,
    kLightTreeNodes,    kCompactToLeaf,     kClusterRoots,        kImportanceHeap,  kLiveBoundMin,
    kLiveBoundMax};

bool IsAmdDevice(ID3D12Device* device)
{
    static int cached = -1;
    if (cached < 0)
    {
        cached = 0;
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) &&
            SUCCEEDED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
        {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            cached = (desc.VendorId == 0x1002) ? 1 : 0; // AMD
        }
    }
    return cached == 1;
}
} // namespace

std::vector<RenderTechnique::DebugView> GuidedPathTracingPass::GetDebugViews() const
{
    return WithBufferViews(BuildDebugViews<GuidingDebugView>(kGuidingDebugViewDocs));
}

bool GuidedPathTracingPass::SetDebugView(int index)
{
    const auto view = magic_enum::enum_cast<GuidingDebugView>(index);
    if (!view)
        return false;
    g_guidingDebugView.Set(*view);
    return true;
}

bool GuidedPathTracingPass::HasActiveDebugView() const
{
    return g_guidingDebugView.Get() != GuidingDebugView::None;
}

bool GuidedPathTracingPass::UseInlineRayQuery()
{
    const int mode = g_inlineRayQuery.Get();
    if (mode == 0) return false;
    if (mode > 0) return true;
    // Auto = pipeline on every vendor: measured dead heat on RDNA (ADR 0011,
    // 824 vs 827 frames/3s), and the pipeline path is the SER-ready one for
    // future Ada+ hardware. The RQ backend stays as an opt-in cross-check.
    return false;
}

void GuidedPathTracingPass::EnsureInlineRayQueryPso()
{
    if (m_inlineRqProgram)
        return;

    m_inlineRqProgram = ShaderProgramCache::Get().GetOrCreateCompute(m_device.Get(), m_globalRootSignature.Get(),
        "resources/shaders/guidedPathTracing.rq.shader", L"GuidedPathTracing InlineRQ PSO");
    spdlog::info("GuidedPathTracingPass: inline-RayQuery compute PSO created");
}

TechniqueDesc GuidedPathTracingPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    // One raygen asset: the compile-time levers suffix it (DxrPass), so a variant
    // no longer needs a sidecar file of its own.
    desc.shaders = {
            {"resources/shaders/guidedPathTracing.rg.shader",    L"GuidedRayGen", ShaderRole::RayGen},
            {"resources/shaders/guidedPathTracing.ms.shader",   L"GuidedMiss",   ShaderRole::Miss},
            {"resources/shaders/raytracing.shadowmiss.shader",  L"ShadowMiss",   ShaderRole::Miss},
            {"resources/shaders/guidedPathTracing.ch.shader",   L"GuidedHit",    ShaderRole::ClosestHit},
            {"resources/shaders/guidedPathTracing.ah.shader",   L"GuidedAnyHit", ShaderRole::AnyHit},
            {"resources/shaders/raytracing.shadowhit.shader",   L"ShadowHit",    ShaderRole::AnyHit},
    };
    // Shadow shaders must land at miss index 1 / hit group index 1 (TraceShadow
    // hardcodes those SBT offsets).
    desc.hitGroups = {
        {L"GuidedHitGroup", L"GuidedHit", L"GuidedAnyHit"},
        {L"ShadowHitGroup", L"",          L"ShadowHit"},
    };
    desc.maxPayloadSize    = 5 * sizeof(uint32_t); // GuidedPayload: 3x uint + float2 (ADR 0007)
    desc.maxAttributeSize  = 2 * sizeof(float);
    // Flat iterative integrator (ADR 0007): every TraceRay — bounce and shadow —
    // launches from raygen; nothing traces from hit/miss shaders.
    desc.maxRecursionDepth = 1;
    return desc;
}

void GuidedPathTracingPass::CreateGlobalRootSignature()
{
    m_globalRootSignature = RootSignatureBuilder(L"GuidedPathTracing GlobalRootSig", /*tableCount*/ 1)
                                .AddFrameLayout()
                                .Add(kGuidedSlots)
                                .WithStaticSamplers()
                                .Build(m_device.Get());

    // Compute programs are keyed on the root signature they were created with;
    // a rebuilt signature (variant reload) must drop them or a later dispatch
    // pairs a stale-layout PSO with the new signature.
    m_inlineRqProgram        = nullptr;
    m_adaptiveQUpdateProgram = nullptr;
}

void GuidedPathTracingPass::EnsureAdaptiveQResources(uint32_t width, uint32_t height)
{
    const uint32_t tilesPerRow    = (width  + ONE_SAMPLE_TILE_SIZE - 1) / ONE_SAMPLE_TILE_SIZE;
    const uint32_t tilesPerColumn = (height + ONE_SAMPLE_TILE_SIZE - 1) / ONE_SAMPLE_TILE_SIZE;
    if (m_tileGuideQ && tilesPerRow == m_tileGridWidth && tilesPerColumn == m_tileGridHeight)
        return;
    m_tileGridWidth  = tilesPerRow;
    m_tileGridHeight = tilesPerColumn;
    const size_t tileCount = size_t(tilesPerRow) * tilesPerColumn;
    m_tileGuideQ = std::make_unique<RWStructuredBuffer<float>>(
        m_device, tileCount, L"GuidedPT TileGuideQ");
    m_tileStrategyStats = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, tileCount * 2, L"GuidedPT TileStrategyStats");
    // Fresh buffers hold undefined data: both shaders treat out-of-range q as
    // 0.5 and the update kernel clears stats after every read, so one frame
    // self-heals — no CPU-side init pass needed.
}

void GuidedPathTracingPass::EnsureAdaptiveQUpdatePso()
{
    if (m_adaptiveQUpdateProgram)
        return;

    m_adaptiveQUpdateProgram = ShaderProgramCache::Get().GetOrCreateCompute(m_device.Get(), m_globalRootSignature.Get(),
        "resources/shaders/vxpgAdaptiveQ.update.shader", L"GuidedPT AdaptiveQ Update PSO");
}

void GuidedPathTracingPass::DeclareDispatchResources(RenderGraph& graph, RenderGraphPassBuilder& dispatchPass)
{
    DeclareVoxelGuidingReads(dispatchPass);

    if (!CompilesLever("onesample"))
        return;

    // The dispatch reads last frame's q and accumulates this frame's stats; the
    // update node below turns one into the other (ADR 0015).
    EnsureAdaptiveQResources(Window::Get().GetWidth(), Window::Get().GetHeight());
    m_tileGuideQHandle        = graph.Import(*m_tileGuideQ, "GuidedPT TileGuideQ");
    m_tileStrategyStatsHandle = graph.Import(*m_tileStrategyStats, "GuidedPT TileStrategyStats");

    dispatchPass.Read(m_tileGuideQHandle, GraphAccess::UnorderedAccessRead);
    dispatchPass.Write(m_tileStrategyStatsHandle, GraphAccess::ComputeWrite);
}

// Every VXPG resource the integrator's signature binds, paired with this frame's
// handle. Driven by the slot table so the two cannot disagree: the hand-written
// version this replaces had drifted into declaring three reads that never happen
// (voxel occupancy, shading points, superpixel centers — none of them bound) and
// omitting the four VPL outputs the raygen actually writes.
void GuidedPathTracingPass::DeclareVoxelGuidingReads(RenderGraphPassBuilder& pass) const
{
    const VxpgGraphHandles& vxpg = *m_frameGuiding;

    const GraphResourceHandle handles[] = {
        vxpg.voxelIrradiance, vxpg.voxelVplCount, vxpg.voxelRepresentative, vxpg.vplPosition, vxpg.superpixelIndex,
        vxpg.vbuffer,         vxpg.clusterVisibilityMask, vxpg.superpixelFuzzyWeight, vxpg.superpixelFuzzyIndex,
        vxpg.counters,        vxpg.compactIds,    vxpg.inverseIndex,       vxpg.voxelFingerprints,
        vxpg.clusterAssignments, vxpg.clusterSeedCompactIds, vxpg.lightTreeNodes, vxpg.lightTreeCompactToLeaf,
        vxpg.lightTreeClusterRoots, vxpg.superpixelClusterHeap, vxpg.liveBoundMin, vxpg.liveBoundMax};
    static_assert(std::size(handles) == std::size(kGuidedGraphSlots), "one handle per graph-visible slot");

    for (size_t i = 0; i < std::size(kGuidedGraphSlots); ++i)
        pass.Declare(kGuidedGraphSlots[i], handles[i]);
}

void GuidedPathTracingPass::AppendPostDispatchNodes(RenderGraph& graph)
{
    if (!CompilesLever("onesample"))
        return;

    // Folds this frame's per-tile strategy stats into the guide-selection
    // probability the NEXT frame's coin reads, then clears the stats. Its consumer
    // is next frame's dispatch, which culling cannot see.
    graph.AddPass("VXPG AdaptiveQ Update",
        [&](RenderGraphPassBuilder& pass)
        {
            pass.NeverCull();
            pass.Read(m_tileStrategyStatsHandle, GraphAccess::UnorderedAccessRead);
            pass.Write(m_tileGuideQHandle, GraphAccess::ComputeWrite);
        },
        [this]()
        {
            EnsureAdaptiveQUpdatePso();
            BindGuidingResources();
            m_commandList->SetPipelineState(m_adaptiveQUpdateProgram->GetPipelineState());
            const uint32_t tileCount = m_tileGridWidth * m_tileGridHeight;
            CommandContext::Get().Dispatch((tileCount + 63) / 64, 1, 1);
        });
}

void GuidedPathTracingPass::BindGuidingResources()
{
    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());
    m_commandList->SetGraphicsRootSignature(nullptr);

    auto* commandList = m_commandList.Get();
    FrameBindingLayout::Bind(commandList, m_globalRootSignature, *m_currentScene, *m_passConstants);

    const RootSignature& rootSignature = m_globalRootSignature;
    rootSignature.Set(commandList, kVoxelGridConstants, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kGuidingCounters, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kGuidingCompactIds, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kGuidingInverseIndex, m_buildPass->GetInverseIndexBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kVoxelFingerprints, m_fingerprintPass->GetVoxelFingerprintsBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kClusterAssignments, m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kClusterSeeds, m_clusterPass->GetClusterSeedCompactIdsBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kLightTreeNodes, m_lightTreePass->GetNodesBufferVA());
    rootSignature.Set(commandList, kCompactToLeaf, m_lightTreePass->GetCompactToLeafBufferVA());
    rootSignature.Set(commandList, kClusterRoots, m_lightTreePass->GetClusterRootsBufferVA());
    rootSignature.Set(commandList, kImportanceHeap, m_lightTreePass->GetSuperpixelClusterHeapBufferVA());
    rootSignature.Set(commandList, kLiveBoundMin, m_buildPass->GetLiveBoundMinBuffer()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kLiveBoundMax, m_buildPass->GetLiveBoundMaxBuffer()->GetGPUVirtualAddress());
    EnsureAdaptiveQResources(Window::Get().GetWidth(), Window::Get().GetHeight());
    rootSignature.Set(commandList, kTileGuideQ, m_tileGuideQ->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kTileStrategyStats, m_tileStrategyStats->GetGPUVirtualAddress());

    const uint32_t tileCount = m_tileGridWidth * m_tileGridHeight;
    rootSignature.SetConstants(commandList, kAdaptiveQConstants, &tileCount, 1);
}

void GuidedPathTracingPass::Render()
{
    if (!m_voxelPass || !m_buildPass || !m_fingerprintPass || !m_clusterPass || !m_lightTreePass)
    {
        spdlog::warn("GuidedPathTracingPass: guiding resources not wired, skipping render");
        return;
    }

    BindGuidingResources();

    if (UseInlineRayQuery())
    {
        // Compute build (ADR 0011): identical bindings/root signature, one
        // thread per pixel, no SBT.
        EnsureInlineRayQueryPso();
        m_commandList->SetPipelineState(m_inlineRqProgram->GetPipelineState());
        const uint32_t width  = Window::Get().GetWidth();
        const uint32_t height = Window::Get().GetHeight();
        CommandContext::Get().Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    }
    else
    {
        D3D12_DISPATCH_RAYS_DESC desc = {};
        desc.RayGenerationShaderRecord.StartAddress = m_shaderBindingTable->GetUnderlyingResource()->GetGPUVirtualAddress();
        desc.RayGenerationShaderRecord.SizeInBytes  = m_shaderBindingTable->GetRayGenSectionSize();

        desc.MissShaderTable.StartAddress  = desc.RayGenerationShaderRecord.StartAddress + desc.RayGenerationShaderRecord.SizeInBytes;
        desc.MissShaderTable.StrideInBytes = m_shaderBindingTable->GetMissEntrySize();
        desc.MissShaderTable.SizeInBytes   = m_shaderBindingTable->GetMissSectionSize();

        desc.HitGroupTable.StartAddress  = desc.MissShaderTable.StartAddress + desc.MissShaderTable.SizeInBytes;
        desc.HitGroupTable.StrideInBytes = m_shaderBindingTable->GetHitEntrySize();
        desc.HitGroupTable.SizeInBytes   = m_shaderBindingTable->GetHitSectionSize();

        desc.Width  = Window::Get().GetWidth();
        desc.Height = Window::Get().GetHeight();
        desc.Depth  = 1;

        m_commandList->SetPipelineState1(m_rtStateObject.Get());
        CommandContext::Get().DispatchRays(desc);
    }
}

REGISTER_TECHNIQUE("Guided Path Tracing (VXPG)", GuidedPathTracingPass)
