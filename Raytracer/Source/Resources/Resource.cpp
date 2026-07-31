#include "pch.h"
#include "Resources/Resource.h"

#include "Resources/ResourceStateTracker.h"
#include "Utils/Utils.h"

bool Resource::CheckFormatSupport(D3D12_FORMAT_SUPPORT1 formatSupport1) const
{
    return (m_formatSupport.Support1 & formatSupport1) != 0;
}

bool Resource::CheckFormatSupport(D3D12_FORMAT_SUPPORT2 formatSupport2) const
{
    return (m_formatSupport.Support2 & formatSupport2) != 0;
}

Resource::Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc,
                   const D3D12_CLEAR_VALUE* clearValue, D3D12_RESOURCE_STATES initialState)
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
        initialState,
        m_clearValue.get(),
        IID_PPV_ARGS(&m_resource)));

    QueryFeatureSupport();
    InitializeTrackedState(initialState);
}

Resource::Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource,
                   const D3D12_CLEAR_VALUE* clearValue, D3D12_RESOURCE_STATES initialState)
    :   m_device(device),
        m_resource(resource)
{
    if (clearValue)
    {
        m_clearValue = std::make_unique<D3D12_CLEAR_VALUE>(*clearValue);
    }

    if (resource)
    {
        QueryFeatureSupport();
        InitializeTrackedState(initialState);
    }
}

Resource::~Resource()
{
    ResourceStateTracker::Get().Unregister(*this);
}

void Resource::InitializeTrackedState(D3D12_RESOURCE_STATES initialState)
{
    const auto desc = m_resource->GetDesc();
    m_trackedState.state              = initialState;
    m_trackedState.isBuffer           = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    m_trackedState.simultaneousAccess = (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0;

    D3D12_HEAP_PROPERTIES heapProperties{};
    D3D12_HEAP_FLAGS heapFlags{};
    if (SUCCEEDED(m_resource->GetHeapProperties(&heapProperties, &heapFlags)))
    {
        m_trackedState.tracked = heapProperties.Type == D3D12_HEAP_TYPE_DEFAULT;
    }

    ResourceStateTracker::Get().Register(*this);
}

void Resource::TransitionChecked(ID3D12GraphicsCommandList* commandList,
                                 D3D12_RESOURCE_STATES expectedBefore,
                                 D3D12_RESOURCE_STATES after)
{
    ResourceStateTracker::Get().TransitionChecked(commandList, *this, expectedBefore, after);
}

void Resource::UavBarrierChecked(ID3D12GraphicsCommandList* commandList)
{
    ResourceStateTracker::Get().UavBarrierChecked(commandList, *this);
}

void Resource::QueryFeatureSupport()
{
    auto desc = m_resource->GetDesc();
    m_formatSupport.Format = desc.Format;
    ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &m_formatSupport, sizeof(m_formatSupport)));
}
