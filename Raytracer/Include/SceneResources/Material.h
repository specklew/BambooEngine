#pragma once

class ConstantBuffer;

struct Material
{
    Material();
    void UpdateMaterial();
    
    std::shared_ptr<ConstantBuffer> m_materialBuffer;

    struct MaterialData
    {
        DirectX::XMFLOAT4 albedoColor = {1.0f, 1.0f, 1.0f, 1.0f};
        // albedo texture <- idk if I should put textures as ids into the buffer? maybe lets just stay with the naive approach and just pack everything into the memory without any shape or form?
        // normal texture
        // roughness texture
        // maybe fresnel?
        // material transform - affine transform of material UVs - will I use it??
    } m_data;

private:
    MaterialData m_mappedData;
};
