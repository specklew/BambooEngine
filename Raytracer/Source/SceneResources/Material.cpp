#include "pch.h"
#include "SceneResources/Material.h"

#include "Renderer.h"
#include "Resources/ConstantBuffer.h"

Material::Material()
{
    Microsoft::WRL::ComPtr<ID3D12Resource> materialResource;
    
    Renderer::g_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(Align(sizeof(MaterialData), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT)),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&materialResource));
    
    m_materialBuffer = std::make_shared<ConstantBuffer>(Renderer::g_device.Get(), materialResource);

    m_materialBuffer->SetResourceName(L"Material Constant Buffer");
}

void Material::UpdateMaterial()
{
    if (m_albedoTexture) m_data.albedo_index = m_albedoTexture->GetTextureIndex();
    if (m_normalTexture) m_data.normal_index = m_normalTexture->GetTextureIndex();
    if (m_metallicRoughnessTexture) m_data.roughness_index = m_metallicRoughnessTexture->GetTextureIndex();
    
    auto p_data = &m_mappedData;
    m_materialBuffer->MapDataToWholeBuffer(reinterpret_cast<void**>(&p_data));
    memcpy(p_data, &m_data, sizeof(MaterialData));
    m_materialBuffer->Unmap();
}
