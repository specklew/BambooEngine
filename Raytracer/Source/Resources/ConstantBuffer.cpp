#include "pch.h"
#include "Resources/ConstantBuffer.h"

ConstantBuffer::ConstantBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource) : Buffer(device, resource)
{
    m_sizeInBytes = GetUnderlyingResource()->GetDesc().Width;
}
