#pragma once
#include "Resource.h"

class Texture : public Resource
{
public:
    Texture(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource);
    int GetTextureIndex() const { return m_textureIndex; }

private:
    int m_textureIndex = -1;
};
