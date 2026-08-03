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
constexpr BindingSlot kVoxelIrradiance = PassTableEntry("gVoxIrradiance", BindingKind::Uav, GUIDED_REG_IRRADIANCE, GlobalDescriptor::VoxelIrradiance);
constexpr BindingSlot kVoxelVplCount = PassTableEntry("gVoxVplCount", BindingKind::Uav, GUIDED_REG_VPL_COUNT, GlobalDescriptor::VoxelVplCount);
constexpr BindingSlot kSuperpixelIndex = PassTableEntry("gSpixelIndexImage", BindingKind::Uav, GUIDED_REG_SUPERPIXEL_INDEX, GlobalDescriptor::SuperpixelIndex);
constexpr BindingSlot kVoxelRepresentative = PassTableEntry("gVoxelRepresentative", BindingKind::Uav, GUIDED_REG_VOXEL_REPRESENTATIVE, GlobalDescriptor::VoxelRepresentative); // debug views 6/7
constexpr BindingSlot kVplPosition = PassTableEntry("gVplPosition", BindingKind::Uav, GUIDED_REG_VPL_POSITION, GlobalDescriptor::VplPosition);
constexpr BindingSlot kVBuffer = PassTableEntry("gVBuffer", BindingKind::Uav, GUIDED_REG_VBUFFER, GlobalDescriptor::VBuffer);
constexpr BindingSlot kVisibilityMask = PassTableEntry("gClusterVisibilityMask", BindingKind::Uav, GUIDED_REG_VISIBILITY_MASK, GlobalDescriptor::ClusterVisibilityMask); // debug view 10
constexpr BindingSlot kFuzzyWeights = PassTableEntry("gFuzzyWeights", BindingKind::Uav, GUIDED_REG_FUZZY_WEIGHT, GlobalDescriptor::FuzzyWeight);
constexpr BindingSlot kFuzzyIndices = PassTableEntry("gFuzzyIndices", BindingKind::Uav, GUIDED_REG_FUZZY_INDEX, GlobalDescriptor::FuzzyIndex);
constexpr BindingSlot kVoxelGridConstants  = PassCbv("VoxelGridCB", REG_VOXEL_GRID_CB);
constexpr BindingSlot kGuidingCounters     = PassUav("gVoxCounters", GUIDED_REG_COUNTERS);
constexpr BindingSlot kGuidingCompactIds   = PassUav("gVoxCompactIds", GUIDED_REG_COMPACT_IDS);
constexpr BindingSlot kGuidingInverseIndex = PassUav("gVoxInverseIndex", GUIDED_REG_INVERSE_INDEX);
constexpr BindingSlot kVoxelFingerprints   = PassUav("gVoxelFingerprints", GUIDED_REG_FINGERPRINTS);
constexpr BindingSlot kClusterAssignments  = PassUav("gVoxelClusterAssignments", GUIDED_REG_CLUSTER_ASSIGNMENTS);
constexpr BindingSlot kClusterSeeds        = PassUav("gClusterSeedCompactIds", GUIDED_REG_CLUSTER_SEEDS);
constexpr BindingSlot kLightTreeNodes      = PassUav("gLightTreeNodes", GUIDED_REG_LIGHT_TREE_NODES);
constexpr BindingSlot kCompactToLeaf       = PassUav("gCompactToLeaf", GUIDED_REG_COMPACT_TO_LEAF);
constexpr BindingSlot kClusterRoots        = PassUav("gClusterRootNodes", GUIDED_REG_CLUSTER_ROOTS);
constexpr BindingSlot kImportanceHeap = PassUav("gSpixelClusterImportanceHeap", GUIDED_REG_IMPORTANCE_HEAP);
constexpr BindingSlot kLiveBoundMin   = PassUav("gVoxelLiveBoundMin", GUIDED_REG_LIVE_BOUND_MIN);
constexpr BindingSlot kLiveBoundMax   = PassUav("gVoxelLiveBoundMax", GUIDED_REG_LIVE_BOUND_MAX);
constexpr BindingSlot kTileGuideQ     = PassUav("gTileGuideQ", GUIDED_REG_TILE_GUIDE_Q);         // ADR 0015
constexpr BindingSlot kTileStrategyStats = PassUav("gTileStrategyStats", GUIDED_REG_TILE_STRATEGY_STATS);
constexpr BindingSlot kAdaptiveQConstants = PassRootConstants("AdaptiveQCB", GUIDED_REG_ADAPTIVE_Q_CB, 1);

constexpr BindingSlot kGuidedSlots[] = {
    kVoxelIrradiance,     kVoxelVplCount,   kSuperpixelIndex,  kVoxelRepresentative, kVplPosition,
    kVBuffer,             kVisibilityMask,  kFuzzyWeights,     kFuzzyIndices,        kVoxelGridConstants,
    kGuidingCounters,     kGuidingCompactIds, kGuidingInverseIndex, kVoxelFingerprints, kClusterAssignments,
    kClusterSeeds,        kLightTreeNodes,  kCompactToLeaf,    kClusterRoots,        kImportanceHeap,
    kLiveBoundMin,        kLiveBoundMax,    kTileGuideQ,       kTileStrategyStats,   kAdaptiveQConstants};

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
    std::vector<DebugView> views;
    for (const GuidingDebugView view : magic_enum::enum_values<GuidingDebugView>())
        views.push_back({static_cast<int>(view), std::string(magic_enum::enum_name(view))});
    return views;
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
    desc.shaders = {
            {m_compileOneSampleMis ? "resources/shaders/guidedPathTracing.rg.onesample.shader"
             : m_compileDebugViews ? "resources/shaders/guidedPathTracing.rg.shader"
                                   : "resources/shaders/guidedPathTracing.rg.clean.shader",
                                                                L"GuidedRayGen", ShaderRole::RayGen},
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
    const uint32_t tilesPerRow    = (width + 15) / 16;
    const uint32_t tilesPerColumn = (height + 15) / 16;
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

    if (!m_compileOneSampleMis)
        return;

    // The dispatch reads last frame's q and accumulates this frame's stats; the
    // update node below turns one into the other (ADR 0015).
    EnsureAdaptiveQResources(Window::Get().GetWidth(), Window::Get().GetHeight());
    m_tileGuideQHandle        = graph.Import(*m_tileGuideQ, "GuidedPT TileGuideQ");
    m_tileStrategyStatsHandle = graph.Import(*m_tileStrategyStats, "GuidedPT TileStrategyStats");

    dispatchPass.Read(m_tileGuideQHandle, GraphAccess::UnorderedAccessRead);
    dispatchPass.Write(m_tileStrategyStatsHandle, GraphAccess::ComputeWrite);
}

// Every VXPG product the integrator samples. This declaration is the only thing
// keeping the twenty-six nodes that produce them alive through culling, so a
// resource dropped from here silently loses its producer rather than erroring.
void GuidedPathTracingPass::DeclareVoxelGuidingReads(RenderGraphPassBuilder& pass) const
{
    constexpr GraphAccess kUavRead = GraphAccess::UnorderedAccessRead;
    const VxpgGraphHandles& vxpg = *m_frameGuiding;

    // Read through the global descriptor table. Voxel occupancy is deliberately
    // absent: no guided shader binds it (the signature has no slot for it, so
    // reflection would reject one that tried). It is a bake product read only by
    // the raster debug views; the bake stays alive through bakedBoundMin/Max,
    // which the guiding build really does read.
    pass.Read(vxpg.voxelIrradiance, kUavRead);
    pass.Read(vxpg.voxelVplCount, kUavRead);
    pass.Read(vxpg.voxelRepresentative, kUavRead);
    pass.Read(vxpg.shadingPoints, kUavRead);
    pass.Read(vxpg.vbuffer, kUavRead);
    pass.Read(vxpg.superpixelIndex, kUavRead);
    pass.Read(vxpg.superpixelCenter, kUavRead);
    pass.Read(vxpg.superpixelFuzzyWeight, kUavRead);
    pass.Read(vxpg.superpixelFuzzyIndex, kUavRead);
    pass.Read(vxpg.clusterVisibilityMask, kUavRead);

    // Read as root UAVs (root parameters 8-19 of this technique's global signature).
    pass.Read(vxpg.counters, kUavRead);
    pass.Read(vxpg.compactIds, kUavRead);
    pass.Read(vxpg.inverseIndex, kUavRead);
    pass.Read(vxpg.voxelFingerprints, kUavRead);
    pass.Read(vxpg.clusterAssignments, kUavRead);
    pass.Read(vxpg.clusterSeedCompactIds, kUavRead);
    pass.Read(vxpg.lightTreeNodes, kUavRead);
    pass.Read(vxpg.lightTreeCompactToLeaf, kUavRead);
    pass.Read(vxpg.lightTreeClusterRoots, kUavRead);
    pass.Read(vxpg.superpixelClusterHeap, kUavRead);
    pass.Read(vxpg.liveBoundMin, kUavRead);
    pass.Read(vxpg.liveBoundMax, kUavRead);
}

void GuidedPathTracingPass::AppendPostDispatchNodes(RenderGraph& graph)
{
    if (!m_compileOneSampleMis)
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
