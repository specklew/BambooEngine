#pragma once
#include "Resource.h"

class Texture : public Resource
{
public:
    // Adopt an existing resource and assign a texture-array slot (glTF material path).
    Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource,
            D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON);

    // Adopt an existing resource whose current state is known (e.g. swap-chain back
    // buffers). No texture-array slot is assigned.
    Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource,
            D3D12_RESOURCE_STATES currentState, const std::wstring& name);

    // Create a committed texture. No texture-array slot is assigned.
    Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc,
            D3D12_RESOURCE_STATES initialState, const std::wstring& name,
            const D3D12_CLEAR_VALUE* clearValue = nullptr);

    int GetTextureIndex() const { return m_textureIndex; }

private:
    int m_textureIndex = -1;
};
