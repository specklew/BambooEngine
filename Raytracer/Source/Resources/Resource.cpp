#include "pch.h"
#include "Resources/Resource.h"

#include "Helpers.h"

bool Resource::CheckFormatSupport(D3D12_FORMAT_SUPPORT1 formatSupport1) const
{
    return (m_formatSupport.Support1 & formatSupport1) != 0;
}

bool Resource::CheckFormatSupport(D3D12_FORMAT_SUPPORT2 formatSupport2) const
{
    return (m_formatSupport.Support2 & formatSupport2) != 0;
}

Resource::Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clearValue)
    :   m_device(device)
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
    
    QueryFeatureSupport();
}

Resource::Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource, const D3D12_CLEAR_VALUE* clearValue)
    :   m_device(device),
        m_resource(resource)
{
    if (clearValue)
    {
        m_clearValue = std::make_unique<D3D12_CLEAR_VALUE>(*clearValue);
    }

    QueryFeatureSupport();
}

void Resource::QueryFeatureSupport()
{
    auto desc = m_resource->GetDesc();
    m_formatSupport.Format = desc.Format;
    ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &m_formatSupport, sizeof(m_formatSupport)));
}
