#pragma once
#include "Constants.h"

// One upload-heap allocation holding NUM_FRAMES copies of a constant buffer.
//
// A single copy was safe only because the frame ended with a full GPU flush: the
// CPU could not reach the next frame's write while the GPU still read this one.
// Frame pacing (ADR 0017 phase 6a) replaces that flush with a wait NUM_FRAMES-1
// frames behind, which is exactly how many copies the CPU can run ahead of — so
// that is how many this holds. The buffer stays mapped for its whole life;
// UPLOAD heap memory is CPU-visible, and mapping per frame bought nothing.
class FrameConstantRing
{
public:
    void Initialize(ID3D12Device* device, size_t sizeInBytes, const wchar_t* debugName);

    void Write(uint32_t frameIndex, const void* data, size_t sizeInBytes);

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS GetGpuVirtualAddress(uint32_t frameIndex) const;

private:
    Microsoft::WRL::ComPtr<ID3D12Resource> m_buffer;
    uint8_t*                               m_mapped = nullptr;
    // Root CBVs take a GPU address, which D3D12 requires to be 256-byte aligned,
    // so the stride is the aligned size rather than the struct size.
    uint64_t                               m_stride = 0;
};
