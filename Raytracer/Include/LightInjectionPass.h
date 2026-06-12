#pragma once

#include "RaytracePass.h"

class VoxelizationPass;

// VXPG light injection (step 2): per frame, traces camera ray + one BSDF
// bounce per pixel, evaluates direct light at the second path vertex and
// atomically injects packed scalar irradiance into the voxel grid.
// Auxiliary pass — not registered as a user-selectable technique.
class LightInjectionPass : public RaytracePass
{
public:
    void SetVoxelizationPass(const std::shared_ptr<VoxelizationPass>& voxelPass) { m_voxelPass = voxelPass; }

    void Render() override;

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;

    // No full-screen output buffer — do not clobber shared heap slot 2
    void CreateRaytracingOutputBuffer() override {}
    void CreateShaderResourceHeap() override;

private:
    std::shared_ptr<VoxelizationPass> m_voxelPass;
};
