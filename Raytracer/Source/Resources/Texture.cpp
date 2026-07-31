#include "pch.h"
#include "Resources/Texture.h"
#include "GlobalDescriptorHeap.h"

Texture::Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource,
                 D3D12_RESOURCE_STATES currentState)
    : Resource(device, resource, nullptr, currentState)
{
    m_textureIndex = GlobalDescriptorHeap::Get().AllocateMaterialTextureSlot();
    std::wstring str = L"Texture " + std::to_wstring(m_textureIndex);
    SetResourceName(str);
}

Texture::Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource,
                 D3D12_RESOURCE_STATES currentState, const std::wstring& name)
    : Resource(device, resource, nullptr, currentState)
{
    SetResourceName(name);
}

Texture::Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc,
                 D3D12_RESOURCE_STATES initialState, const std::wstring& name,
                 const D3D12_CLEAR_VALUE* clearValue)
    : Resource(device, desc, clearValue, initialState)
{
    SetResourceName(name);
}
