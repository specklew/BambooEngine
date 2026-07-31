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
    // area light with this constant radiance.
    DirectX::XMFLOAT3 m_emissiveRadiance{ 0.0f, 0.0f, 0.0f };

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
