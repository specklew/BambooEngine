#pragma once
#include "Resource.h"

class Buffer : public Resource
{
public:
    size_t GetBufferSize() const { return m_resource->GetDesc().Width; }
protected:
    Buffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc);
    Buffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource);
    
};
