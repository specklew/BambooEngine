#pragma once

#include "DxrPass.h"

// Shared primary-visibility buffer pass (ADR 0004, SIByL raytraced-vbuffer):
// per frame, traces one jittered camera ray per pixel and stores the hit's
// identity (instance + primitive + barycentrics) as RGBA32_UINT. Light
// injection and the guided integrator reconstruct their first path vertex
// from it instead of tracing their own primaries.
// Auxiliary pass — not registered as a user-selectable technique.
class VBufferPass : public DxrPass
{
public:
    void Render() override;

    Microsoft::WRL::ComPtr<ID3D12Resource> GetVBufferTexture() const { return m_vbufferTex; }

protected:
    TechniqueDesc GetTechniqueDesc() const override;
    void CreateGlobalRootSignature() override;
    void CreateShaderResourceHeap() override;

private:
    void CreateVBufferResource();

    Microsoft::WRL::ComPtr<ID3D12Resource> m_vbufferTex;
};
