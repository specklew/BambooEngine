#pragma once
#include "Resources/Texture.h"

class ConstantBuffer;

struct Material
{
    Material();
    void UpdateMaterial();
    
    std::shared_ptr<ConstantBuffer> m_materialBuffer;
    std::shared_ptr<Texture> m_albedoTexture;

    struct MaterialData
    {
        DirectX::XMFLOAT4 albedoColor = {1.0f, 1.0f, 1.0f, 1.0f};
        uint32_t albedo_index = -1;
        uint32_t normal_index = -1;
        uint32_t roughness_index = -1;
        uint32_t padding = -1;
        // albedo texture <- idk if I should put textures as ids into the buffer? maybe lets just stay with the naive approach and just pack everything into the memory without any shape or form?
        // normal texture
        // roughness texture
        // maybe fresnel?
        // material transform - affine transform of material UVs - will I use it??
    } m_data;

private:
    MaterialData m_mappedData;
};
