#include "pch.h"
#include "Utils/PassConstants.h"

#include "Renderer.h"
#include "Resources/ConstantBuffer.h"

PassConstants::PassConstants()
{
    using namespace Microsoft::WRL;

    ComPtr<ID3D12Resource> resource;

    const HRESULT hr = Renderer::g_d3d12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(sizeof(MappedData)),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&resource));

    ThrowIfFailed(hr);

    m_buffer = std::make_unique<ConstantBuffer>(Renderer::g_d3d12Device, resource);
}

void PassConstants::Map()
{
    auto pData = &m_mappedData;
    m_buffer->MapDataToWholeBuffer(reinterpret_cast<void**>(&pData));

    memcpy(pData, &data, sizeof(MappedData));
    
    m_buffer->Unmap();
}

D3D12_GPU_VIRTUAL_ADDRESS PassConstants::GetGpuVirtualAddress() const
{
    return m_buffer->GetUnderlyingResource()->GetGPUVirtualAddress();
}
