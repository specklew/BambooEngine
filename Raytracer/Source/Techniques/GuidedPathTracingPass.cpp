#include "pch.h"
#include "CommandContext.h"
#include "Techniques/GuidedPathTracingPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "FrameBindingLayout.h"
#include "GlobalDescriptorHeap.h"
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
// Pass-scoped root parameters, appended after the frame prefix so both the
// signature and the bind call read from one list.
enum GuidedRootParameter : uint32_t
{
    VoxelGridConstantsCbv = FrameBindingLayout::kPassRootParameterStart,
    GuidingCountersUav,
    GuidingCompactIdsUav,
    GuidingInverseIndexUav,
    VoxelFingerprintsUav,
    ClusterAssignmentsUav,
    ClusterSeedsUav,
    LightTreeNodesUav,
    CompactToLeafUav,
    ClusterRootsUav,
    ImportanceHeapUav,
    LiveBoundMinUav,
    LiveBoundMaxUav,
    TileGuideQUav,
    TileStrategyStatsUav,
    GuidedRootParameterCount
};

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
    // Base 7-param scene binding extended with: voxel irradiance/count UAVs
    // (u1/u2, global heap slots 520/521), voxel grid CBV (b4), the superpixel
    // index texture (u5, heap slot 523), and root UAVs for the guiding
    // distribution and light-tree buffers (u3 counters, u4 ids, u6 inverse
    // index, u10-u17 fingerprint/cluster/tree/heap).

    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    FrameBindingLayout::AppendFrameRanges(ranges);

    D3D12_DESCRIPTOR_RANGE voxelIrradianceRange;
    voxelIrradianceRange.BaseShaderRegister = 1;
    voxelIrradianceRange.NumDescriptors = 1;
    voxelIrradianceRange.RegisterSpace = 0;
    voxelIrradianceRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelIrradianceRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelIrradiance);

    D3D12_DESCRIPTOR_RANGE voxelVplCountRange;
    voxelVplCountRange.BaseShaderRegister = 2;
    voxelVplCountRange.NumDescriptors = 1;
    voxelVplCountRange.RegisterSpace = 0;
    voxelVplCountRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelVplCountRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelVplCount);

    // Debug views 6/7 read the injection-pass outputs (texture UAVs can't be
    // root descriptors, so they ride the shared-heap table at their slots).
    D3D12_DESCRIPTOR_RANGE voxelRepresentativeRange;
    voxelRepresentativeRange.BaseShaderRegister = 7;
    voxelRepresentativeRange.NumDescriptors = 1;
    voxelRepresentativeRange.RegisterSpace = 0;
    voxelRepresentativeRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelRepresentativeRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelRepresentative);

    D3D12_DESCRIPTOR_RANGE vplPositionRange;
    vplPositionRange.BaseShaderRegister = 8;
    vplPositionRange.NumDescriptors = 1;
    vplPositionRange.RegisterSpace = 0;
    vplPositionRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    vplPositionRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VplPosition);

    D3D12_DESCRIPTOR_RANGE vbufferRange;
    vbufferRange.BaseShaderRegister = 9; // u9
    vbufferRange.NumDescriptors = 1;
    vbufferRange.RegisterSpace = 0;
    vbufferRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    vbufferRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VBuffer);

    // Cluster-visibility mask (debug view 10), texture UAV in the shared heap.
    D3D12_DESCRIPTOR_RANGE clusterMaskRange;
    clusterMaskRange.BaseShaderRegister = 13; // u13
    clusterMaskRange.NumDescriptors = 1;
    clusterMaskRange.RegisterSpace = 0;
    clusterMaskRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    clusterMaskRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::ClusterVisibilityMask);

    // Superpixel index texture (SLIC assignment) — selects the top-level heap
    // row for both MIS strategies.
    D3D12_DESCRIPTOR_RANGE spixelIndexRange;
    spixelIndexRange.BaseShaderRegister = 5; // u5
    spixelIndexRange.NumDescriptors = 1;
    spixelIndexRange.RegisterSpace = 0;
    spixelIndexRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    spixelIndexRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::SuperpixelIndex);

    // Fuzzy 4-nearest blend (superpixel pass outputs): per-pixel weights + ids
    // for the guided integrator's mixture top-level pdf.
    D3D12_DESCRIPTOR_RANGE fuzzyWeightRange;
    fuzzyWeightRange.BaseShaderRegister = 20; // u20
    fuzzyWeightRange.NumDescriptors = 1;
    fuzzyWeightRange.RegisterSpace = 0;
    fuzzyWeightRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    fuzzyWeightRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::FuzzyWeight);

    D3D12_DESCRIPTOR_RANGE fuzzyIndexRange;
    fuzzyIndexRange.BaseShaderRegister = 21; // u21
    fuzzyIndexRange.NumDescriptors = 1;
    fuzzyIndexRange.RegisterSpace = 0;
    fuzzyIndexRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    fuzzyIndexRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::FuzzyIndex);

    for (const D3D12_DESCRIPTOR_RANGE& range :
         {voxelIrradianceRange, voxelVplCountRange, voxelRepresentativeRange, vplPositionRange, vbufferRange,
          clusterMaskRange, spixelIndexRange, fuzzyWeightRange, fuzzyIndexRange})
        ranges.push_back(range);

    CD3DX12_ROOT_PARAMETER rootParameters[GuidedRootParameterCount];
    FrameBindingLayout::FillFrameRootParameters(rootParameters, ranges);
    rootParameters[VoxelGridConstantsCbv].InitAsConstantBufferView(4, 0);   // Voxel grid constants
    rootParameters[GuidingCountersUav].InitAsUnorderedAccessView(3, 0);     // Guiding counters
    rootParameters[GuidingCompactIdsUav].InitAsUnorderedAccessView(4, 0);   // Guiding compact ids
    rootParameters[GuidingInverseIndexUav].InitAsUnorderedAccessView(6, 0); // Guiding inverse index
    rootParameters[VoxelFingerprintsUav].InitAsUnorderedAccessView(10, 0);  // Voxel fingerprints (debug view 8)
    rootParameters[ClusterAssignmentsUav].InitAsUnorderedAccessView(11, 0); // Cluster assignments (sampling + view 9)
    rootParameters[ClusterSeedsUav].InitAsUnorderedAccessView(12, 0);       // Cluster seeds (debug view 9)
    rootParameters[LightTreeNodesUav].InitAsUnorderedAccessView(14, 0);     // Light tree nodes (sampling + view 11)
    rootParameters[CompactToLeafUav].InitAsUnorderedAccessView(15, 0);      // Compact->leaf map (sampling + view 11)
    rootParameters[ClusterRootsUav].InitAsUnorderedAccessView(16, 0);       // Cluster root nodes (sampling + view 11)
    rootParameters[ImportanceHeapUav].InitAsUnorderedAccessView(17, 0);     // Top-level importance heap (view 12)
    rootParameters[LiveBoundMinUav].InitAsUnorderedAccessView(18, 0);       // Live voxel bound min (guide sampling)
    rootParameters[LiveBoundMaxUav].InitAsUnorderedAccessView(19, 0);       // Live voxel bound max (guide sampling)
    rootParameters[TileGuideQUav].InitAsUnorderedAccessView(22, 0);         // Per-tile adaptive q (ADR 0015)
    rootParameters[TileStrategyStatsUav].InitAsUnorderedAccessView(23, 0);  // Per-tile strategy stats (ADR 0015)

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(_countof(rootParameters), rootParameters);

    auto static_samplers = Renderer::GetStaticSamplers();
    rootSignatureDesc.NumStaticSamplers = static_cast<UINT>(static_samplers.size());
    rootSignatureDesc.pStaticSamplers   = static_samplers.data();
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    m_globalRootSignature = RootSignatureLibrary::Get().Create(m_device.Get(), rootSignatureDesc,
                                                               L"GuidedPathTracing GlobalRootSig", true);

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

void GuidedPathTracingPass::Render()
{
    if (!m_voxelPass || !m_buildPass || !m_fingerprintPass || !m_clusterPass || !m_lightTreePass)
    {
        spdlog::warn("GuidedPathTracingPass: guiding resources not wired, skipping render");
        return;
    }

    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());
    m_commandList->SetGraphicsRootSignature(nullptr);

    FrameBindingLayout::BindFrameRootParameters(m_commandList.Get(), *m_currentScene, *m_passConstants);

    m_commandList->SetComputeRootConstantBufferView(VoxelGridConstantsCbv, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(GuidingCountersUav, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(GuidingCompactIdsUav, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(GuidingInverseIndexUav, m_buildPass->GetInverseIndexBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(VoxelFingerprintsUav, m_fingerprintPass->GetVoxelFingerprintsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(ClusterAssignmentsUav, m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(ClusterSeedsUav, m_clusterPass->GetClusterSeedCompactIdsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(LightTreeNodesUav, m_lightTreePass->GetNodesBufferVA());
    m_commandList->SetComputeRootUnorderedAccessView(CompactToLeafUav, m_lightTreePass->GetCompactToLeafBufferVA());
    m_commandList->SetComputeRootUnorderedAccessView(ClusterRootsUav, m_lightTreePass->GetClusterRootsBufferVA());
    m_commandList->SetComputeRootUnorderedAccessView(ImportanceHeapUav, m_lightTreePass->GetSuperpixelClusterHeapBufferVA());
    m_commandList->SetComputeRootUnorderedAccessView(LiveBoundMinUav, m_buildPass->GetLiveBoundMinBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(LiveBoundMaxUav, m_buildPass->GetLiveBoundMaxBuffer()->GetGPUVirtualAddress());
    EnsureAdaptiveQResources(Window::Get().GetWidth(), Window::Get().GetHeight());
    m_commandList->SetComputeRootUnorderedAccessView(TileGuideQUav, m_tileGuideQ->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(TileStrategyStatsUav, m_tileStrategyStats->GetGPUVirtualAddress());

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

    // Adaptive-q update (ADR 0015): fold this frame's per-tile strategy stats
    // into the guide-selection probability the NEXT frame's coin reads, then
    // clear the stats. Root bindings persist from the dispatch above (same
    // compute root signature).
    if (m_compileOneSampleMis)
    {
        EnsureAdaptiveQUpdatePso();
        m_tileStrategyStats->UavBarrier(m_commandList.Get());
        m_commandList->SetPipelineState(m_adaptiveQUpdateProgram->GetPipelineState());
        const uint32_t tileCount = m_tileGridWidth * m_tileGridHeight;
        CommandContext::Get().Dispatch((tileCount + 63) / 64, 1, 1);
        m_tileGuideQ->UavBarrier(m_commandList.Get());
    }

}

REGISTER_RAYTRACE_TECHNIQUE("Guided Path Tracing (VXPG)", GuidedPathTracingPass)
