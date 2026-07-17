#pragma once

#include "RaytracePass.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgFingerprintPass;
class VxpgClusterPass;
class VxpgLightTreePass;

// VXPG guided path tracing technique. Two-sample MIS at the first bounce
// between BSDF sampling and the voxel irradiance distribution (CDF over
// compacted voxels, cone sampling toward the chosen voxel). Falls back to a
// uniform-sphere guide when no guiding data exists.
class GuidedPathTracingPass : public RaytracePass
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
    VxpgStage RequiredVxpgStage() const override { return VxpgStage::LightTree; }

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;

private:
    // Inline-RayQuery compute build of the integrator (ADR 0011), created
    // lazily on first use; shares the global root signature and bindings.
    bool UseInlineRayQuery();
    void EnsureInlineRayQueryPso();
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_inlineRqPso;

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass> m_buildPass;
    std::shared_ptr<VxpgFingerprintPass>   m_fingerprintPass;
    std::shared_ptr<VxpgClusterPass>       m_clusterPass;
    std::shared_ptr<VxpgLightTreePass>     m_lightTreePass;
};
