#pragma once
#include "Buffer.h"
#include "Utils/Utils.h"

// GPU read-write structured buffer — mirrors HLSL RWStructuredBuffer<T>.
// DEFAULT heap with ALLOW_UNORDERED_ACCESS, created in COMMON (see
// RenderingUtils::CreateUavBuffer). Bound as a root UAV via GetGPUVirtualAddress;
// no descriptor/heap slot is allocated (add a view type if a pass ever needs one).
template <typename T>
class RWStructuredBuffer : public Buffer
{
public:
    RWStructuredBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, size_t elementCount, const wchar_t* name = nullptr)
        : Buffer(device, RenderingUtils::CreateUavBuffer(device.Get(), elementCount * sizeof(T), name)),
          m_elementsCount(elementCount) {}

    size_t GetElementsCount() const { return m_elementsCount; }

    D3D12_GPU_VIRTUAL_ADDRESS GetGPUVirtualAddress() const { return m_resource->GetGPUVirtualAddress(); }

    void UavBarrier(ID3D12GraphicsCommandList* commandList) const
    {
        auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_resource.Get());
        commandList->ResourceBarrier(1, &barrier);
    }

private:
    size_t m_elementsCount;
};
