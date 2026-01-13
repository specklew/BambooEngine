#include "pch.h"
#include "SceneResources/Material.h"

#include "Renderer.h"
#include "Resources/ConstantBuffer.h"

Material::Material()
{
    Microsoft::WRL::ComPtr<ID3D12Resource> materialResource;
    
    Renderer::g_d3d12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(128),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&materialResource));
    
    m_materialBuffer = std::make_shared<ConstantBuffer>(Renderer::g_d3d12Device.Get(), materialResource);

    m_materialBuffer->SetResourceName(L"Material Constant Buffer");

    m_pData = &m_data;
    m_materialBuffer->MapDataToWholeBuffer(reinterpret_cast<void**>(&m_pData));
}
