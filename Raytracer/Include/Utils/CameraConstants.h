#pragma once
#include "Utils/FrameConstantRing.h"

// The camera matrices every pass binds, in the layout `cbuffer CameraParams` in
// FrameBindings.hlsl declares. Frame state, exactly like PassConstants: written
// once per frame by the Renderer and read by every pass that binds the frame
// layout, the rasterization technique and the cluster-visibility pass.
//
// A singleton for the same reason CommandContext and GlobalDescriptorHeap are:
// there is one per frame, every pass needs it, and threading it through each
// pass's Initialize would say nothing the name does not already say.
//
// It is a ROOT CBV rather than a global-heap descriptor. That is what makes the
// ring possible at all: a root CBV is bound by GPU address, so the frame picks
// its own copy at bind time, whereas one heap descriptor can only point at one
// copy and rewriting it per frame would race the frames still reading it.
class CameraConstants
{
public:
    static CameraConstants& Get();

    void Initialize(ID3D12Device* device);

    struct MappedData
    {
        DirectX::XMFLOAT4X4 worldViewProj;
        DirectX::XMFLOAT4X4 view;
        DirectX::XMFLOAT4X4 projection;
        DirectX::XMFLOAT4X4 viewInverse;
        DirectX::XMFLOAT4X4 projectionInverse;
    };

    // Publishes this frame's matrices into the frame's own copy, and is what
    // GetGpuVirtualAddress() answers for until the next frame calls it.
    void Update(uint32_t frameIndex, const MappedData& matrices);

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS GetGpuVirtualAddress() const;

private:
    FrameConstantRing m_ring;
    uint32_t          m_frameIndex = 0;
};
