#include "pch.h"
#include "Resources/Texture.h"
#include "Renderer.h"

Texture::Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource) : Resource(device, resource)
{
    m_textureIndex = Renderer::g_textureIndex++;
    std::wstring str = L"Texture " + std::to_wstring(m_textureIndex);
    SetResourceName(str);
}
