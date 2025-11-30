#include "pch.h"
#include "Resources/ConstantBuffer.h"

ConstantBuffer::ConstantBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource) : Buffer(device, resource)
{
    m_sizeInBytes = GetUnderlyingResource()->GetDesc().Width;
}

void ConstantBuffer::MapDataToWholeBuffer(DirectX::XMFLOAT4X4** data) const
{
    GetUnderlyingResource()->Map(0, nullptr, reinterpret_cast<void**>(data));
}

void ConstantBuffer::Unmap() const
{
    GetUnderlyingResource()->Unmap(0, nullptr);
}
