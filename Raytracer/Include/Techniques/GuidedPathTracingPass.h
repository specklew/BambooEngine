#pragma once

#include "ShaderProgram.h"

#include "DxrTechnique.h"
#include "Resources/RWStructuredBuffer.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgFingerprintPass;
class VxpgClusterPass;
class VxpgLightTreePass;

// VXPG guided path tracing technique. Two-sample MIS at the first bounce between
// BSDF sampling and the tree-backed voxel guide: per-superpixel importance heap
// -> cluster-root light-tree walk -> exact solid-angle sampling of the chosen
// voxel's visible AABB faces. A pixel with no guiding data (no lit voxels, or
// every fuzzy parent's heap empty) is BSDF-only at full MIS weight — there is no
// uniform-sphere fallback. See guidedPathTracing.hlsl for the full chain.
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

    // Every root binding the guided dispatch needs.
    void BindGuidingResources();

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass> m_buildPass;
    std::shared_ptr<VxpgFingerprintPass>   m_fingerprintPass;
    std::shared_ptr<VxpgClusterPass>       m_clusterPass;
    std::shared_ptr<VxpgLightTreePass>     m_lightTreePass;
};
