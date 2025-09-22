#include "pch.h"
#include "Resource.h"

#include "Helpers.h"

Resource::Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clearValue)
    : m_device(device)
{
    if (clearValue)
    {
        m_clearValue = std::make_unique<D3D12_CLEAR_VALUE>(*clearValue);
    }

    ThrowIfFailed(m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        m_clearValue.get(),
        IID_PPV_ARGS(&m_resource)));

    QueryFormatSupport();
}

Resource::Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device,
    const Microsoft::WRL::ComPtr<ID3D12Resource>& resource, const D3D12_CLEAR_VALUE* clearValue)
{
}

void Resource::QueryFormatSupport()
{
}
