#pragma once

#include "DxrPass.h"
#include "RenderGraph.h"
#include "VxpgGraphHandles.h"

class VoxelizationPass;
class GpuMemoryReport;

// VXPG light injection (step 2): per frame, traces camera ray + one BSDF
// bounce per pixel, evaluates direct light at the second path vertex and
// atomically injects packed scalar irradiance into the voxel grid.
// Auxiliary pass — not registered as a user-selectable technique.
class LightInjectionPass : public DxrPass
{
public:
    // P5: the resources this stage holds, for the guiding chain's memory report.
    // Declared per pass rather than collected from a base class because ADR 0017 left
    // most of this chain as raw ComPtr, which no base class ever sees.
    void ReportMemory(GpuMemoryReport& report) const;
    void SetVoxelizationPass(const std::shared_ptr<VoxelizationPass>& voxelPass) { m_voxelPass = voxelPass; }

    void Render() override;

    // Everything this pass's node touches, taken from the slot table so the two
    // descriptions cannot drift (ADR 0017 step 3).
    void DeclareGraphResources(RenderGraphPassBuilder& pass, const VxpgGraphHandles& vxpg) const;

    // Primary-hit G-buffer (worldPos + octahedral normal), consumed by superpixel clustering.
    Microsoft::WRL::ComPtr<ID3D12Resource> GetShadingPointsTexture() const { return m_shadingPointsTex; }

    // VXPG B+: per-voxel representative VPL (pos + octa normal, Texture3D), consumed by the
    // fingerprint pass; per-pixel VPL hit position (Texture2D), consumed by cvis assignment.
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVoxelRepresentativeTexture() const { return m_voxelRepresentativeTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplPositionTexture() const { return m_vplPositionTex; }
    // The injected sample in the form the guided integrator can reuse as its own
    // BSDF MIS sample (vxpg.injection.reuseInMis).
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplRadianceTexture() const { return m_vplRadianceTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplEmitterTexture() const { return m_vplEmitterTex; }

    // Recreates the grid-sized representative texture after a voxel-grid resize.
    // Caller must have flushed the GPU first (the old texture may be in flight).
    void OnVoxelGridResize() { CreateRepresentativeResources(); }

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;
    void CreateShaderResourceHeap() override;


private:
    void CreateShadingPointsResource();
    void CreateRepresentativeResources();

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_shadingPointsTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_voxelRepresentativeTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vplPositionTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vplRadianceTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vplEmitterTex;
};
