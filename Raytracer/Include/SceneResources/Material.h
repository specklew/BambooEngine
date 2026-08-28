#pragma once
#include "Resources/Texture.h"

class ConstantBuffer;

struct Material
{
    Material();
    void UpdateMaterial();
    
    std::shared_ptr<ConstantBuffer> m_materialBuffer;
    std::shared_ptr<Texture> m_albedoTexture;
    std::shared_ptr<Texture> m_normalTexture;
    std::shared_ptr<Texture> m_metallicRoughnessTexture;
    // CPU-only (raster untouched): >0 marks every triangle of the primitive an
    // area light. glTF emission is emissiveFactor * emissiveStrength * emissiveTexture,
    // so this holds the constant factor and the texture modulates it per texel.
    DirectX::XMFLOAT3 m_emissiveRadiance{ 0.0f, 0.0f, 0.0f };
    std::shared_ptr<Texture> m_emissiveTexture;
    // Whole-image mean of m_emissiveTexture in linear space, multiplicative identity
    // when there is no texture. Light-pool selection weight only.
    DirectX::XMFLOAT3 m_emissiveTextureAverage{ 1.0f, 1.0f, 1.0f };

    struct MaterialData
    {
        DirectX::XMFLOAT4 baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
        int albedo_index = -1;
        int normal_index = -1;
        int roughness_index = -1;
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        bool isOpaque{true};
    } m_data;

private:
    MaterialData m_mappedData;
};
