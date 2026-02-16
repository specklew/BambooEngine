#pragma once
#include "Buffer.h"

template <typename T>
class StructuredBuffer : public Buffer
{
public:
    StructuredBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource, const size_t elementCount)
    : Buffer(device, resource), m_elementsCount(elementCount) {}

    size_t GetElementsCount() const { return m_elementsCount; }
    
private:
    Microsoft::WRL::ComPtr<ID3D12Resource> m_uploadResource;
    size_t m_elementsCount;
};