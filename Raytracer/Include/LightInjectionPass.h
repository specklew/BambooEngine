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

    // Primary-hit G-buffer (worldPos + octahedral normal), consumed by superpixel clustering.
    Microsoft::WRL::ComPtr<ID3D12Resource> GetShadingPointsTexture() const { return m_shadingPointsTex; }

    // VXPG B+: per-voxel representative VPL (pos + octa normal, Texture3D), consumed by the
    // fingerprint pass; per-pixel VPL hit position (Texture2D), consumed by cvis assignment.
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVoxelRepresentativeTexture() const { return m_voxelRepresentativeTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplPositionTexture() const { return m_vplPositionTex; }

    // Recreates the grid-sized representative texture after a voxel-grid resize.
    // Caller must have flushed the GPU first (the old texture may be in flight).
    void OnVoxelGridResize() { CreateRepresentativeResources(); }

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;

    // No full-screen output buffer — do not clobber shared heap slot 2
    void CreateRaytracingOutputBuffer() override {}
    void CreateShaderResourceHeap() override;

private:
    void CreateShadingPointsResource();
    void CreateRepresentativeResources();

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_shadingPointsTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_voxelRepresentativeTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vplPositionTex;
};
