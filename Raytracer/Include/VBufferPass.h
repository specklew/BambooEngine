#pragma once

#include "DxrPass.h"
#include "RenderGraph.h"
#include "VxpgGraphHandles.h"
class GpuMemoryReport;

// Shared primary-visibility buffer pass (ADR 0004, SIByL raytraced-vbuffer):
// per frame, traces one jittered camera ray per pixel and stores the hit's
// identity (instance + primitive + barycentrics) as RGBA32_UINT. Light
// injection and the guided integrator reconstruct their first path vertex
// from it instead of tracing their own primaries.
// Auxiliary pass — not registered as a user-selectable technique.
class VBufferPass : public DxrPass
{
public:
    // P5: the resources this stage holds, for the guiding chain's memory report.
    // Declared per pass rather than collected from a base class because ADR 0017 left
    // most of this chain as raw ComPtr, which no base class ever sees.
    void ReportMemory(GpuMemoryReport& report) const;
    void Render() override;

    // Everything this pass's node touches, taken from the slot table so the two
    // descriptions cannot drift (ADR 0017 step 3).
    void DeclareGraphResources(RenderGraphPassBuilder& pass, const VxpgGraphHandles& vxpg) const;

    Microsoft::WRL::ComPtr<ID3D12Resource> GetVBufferTexture() const { return m_vbufferTex; }

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;
    void CreateShaderResourceHeap() override;


private:
    void CreateVBufferResource();

    Microsoft::WRL::ComPtr<ID3D12Resource> m_vbufferTex;
};
