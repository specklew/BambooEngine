#include "pch.h"
#include "Utils/FrameConstantRing.h"

#include "Utils/Utils.h"

void FrameConstantRing::Initialize(ID3D12Device* device, size_t sizeInBytes, const wchar_t* debugName)
{
    m_stride = Align(sizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto bufferDesc     = CD3DX12_RESOURCE_DESC::Buffer(m_stride * Constants::Graphics::NUM_FRAMES);
    ThrowIfFailed(device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                  IID_PPV_ARGS(&m_buffer)));
    m_buffer->SetName(debugName);

    // The GPU never writes here, so the read range is empty.
    const CD3DX12_RANGE readRange(0, 0);
    ThrowIfFailed(m_buffer->Map(0, &readRange, reinterpret_cast<void**>(&m_mapped)));
}

void FrameConstantRing::Write(uint32_t frameIndex, const void* data, size_t sizeInBytes)
{
    assert(m_mapped && "FrameConstantRing written before Initialize");
    assert(sizeInBytes <= m_stride && "FrameConstantRing write is larger than the size it was created for");
    std::memcpy(m_mapped + m_stride * frameIndex, data, sizeInBytes);
}

D3D12_GPU_VIRTUAL_ADDRESS FrameConstantRing::GetGpuVirtualAddress(uint32_t frameIndex) const
{
    assert(m_buffer && "FrameConstantRing bound before Initialize");
    return m_buffer->GetGPUVirtualAddress() + m_stride * frameIndex;
}
