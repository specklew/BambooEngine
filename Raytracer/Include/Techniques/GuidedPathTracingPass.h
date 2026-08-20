#pragma once

#include "ShaderProgram.h"

#include "DxrTechnique.h"
#include "Resources/RWStructuredBuffer.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgFingerprintPass;
class VxpgClusterPass;
class VxpgLightTreePass;

// VXPG guided path tracing technique. Two-sample MIS at the first bounce
// between BSDF sampling and the voxel irradiance distribution (CDF over
// compacted voxels, cone sampling toward the chosen voxel). Falls back to a
// uniform-sphere guide when no guiding data exists.
class GuidedPathTracingPass : public DxrTechnique
{
public:
    // Wired by the Renderer after construction (registry factory takes no args)
    void SetGuidingResources(
        const std::shared_ptr<VoxelizationPass>& voxelPass,
        const std::shared_ptr<VoxelGuidingBuildPass>& buildPass,
        const std::shared_ptr<VxpgFingerprintPass>& fingerprintPass,
        const std::shared_ptr<VxpgClusterPass>& clusterPass,
        const std::shared_ptr<VxpgLightTreePass>& lightTreePass)
    {
        m_voxelPass = voxelPass;
        m_buildPass = buildPass;
        m_fingerprintPass = fingerprintPass;
        m_clusterPass = clusterPass;
        m_lightTreePass = lightTreePass;
    }

    void Render() override;

    // Consumes voxelize -> inject -> guiding distribution -> fingerprint ->
    // cluster -> cluster-visibility. Fingerprint/cluster/cvis are required always
    // (not just for debug views 8/9/10): the tree passes consume them and their
    // cost belongs in equal-time benchmarks.
    bool UsesVoxelGuiding() const override { return true; }

    // The BSDF subtree of this integrator writes the VPLs (ADR 0009).
    bool ProducesGuidingVpls() const override { return true; }

    // This technique reads GuidingDebugView, not the raytracing enum DxrTechnique
    // offers, so the dropdown and headless enumeration see the guide's own views.
    std::vector<DebugView> GetDebugViews() const override;
    bool SetDebugView(int index) override;
    bool HasActiveDebugView() const override;
    int  GetDebugMode() const override { return 0; } // guided views ride guidingFlags, not debugMode

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;

    void DeclareDispatchResources(RenderGraph& graph, RenderGraphPassBuilder& dispatchPass) override;
    void AppendPostDispatchNodes(RenderGraph& graph) override;

private:
    void DeclareVoxelGuidingReads(RenderGraphPassBuilder& pass) const;

private:
    // Inline-RayQuery compute build of the integrator (ADR 0011), created
    // lazily on first use; shares the global root signature and bindings.
    bool UseInlineRayQuery();
    void EnsureInlineRayQueryPso();
    ComputeProgram* m_inlineRqProgram = nullptr;
    std::string     m_inlineRqVariantKey; // lever key the RQ PSO was compiled with (ADR 0020)

    // One-sample MIS adaptive q (ADR 0015): per-16x16-tile guide-selection
    // probability + this frame's per-strategy luminance stats, folded into q
    // by a small compute update after every guided dispatch.
    void EnsureAdaptiveQResources(uint32_t width, uint32_t height);
    void EnsureAdaptiveQUpdatePso();
    std::unique_ptr<RWStructuredBuffer<float>>    m_tileGuideQ;
    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_tileStrategyStats;
    ComputeProgram* m_adaptiveQUpdateProgram = nullptr;
    uint32_t m_tileGridWidth  = 0;
    uint32_t m_tileGridHeight = 0;
    // Handed from the dispatch declaration to the update node; imports last one
    // frame, so these are refreshed every time the graph is rebuilt.
    GraphResourceHandle m_tileGuideQHandle       = InvalidGraphResource;
    GraphResourceHandle m_tileStrategyStatsHandle = InvalidGraphResource;

    // Every root binding the guided dispatch needs, shared with the adaptive-q
    // update node so neither depends on the other having run first.
    void BindGuidingResources();

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass> m_buildPass;
    std::shared_ptr<VxpgFingerprintPass>   m_fingerprintPass;
    std::shared_ptr<VxpgClusterPass>       m_clusterPass;
    std::shared_ptr<VxpgLightTreePass>     m_lightTreePass;
};
