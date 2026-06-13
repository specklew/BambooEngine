#pragma once

#include "RaytracePass.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;

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
        const std::shared_ptr<VoxelGuidingBuildPass>& buildPass)
    {
        m_voxelPass = voxelPass;
        m_buildPass = buildPass;
    }

    void Render() override;

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;

private:
    std::shared_ptr<VoxelizationPass>     m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass> m_buildPass;
};
