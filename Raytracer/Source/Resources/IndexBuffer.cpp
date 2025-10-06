#include "pch.h"
#include "Resources/IndexBuffer.h"

IndexBuffer::IndexBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, size_t indexCount, DXGI_FORMAT format)
    :   Buffer(device, CD3DX12_RESOURCE_DESC::Buffer(indexCount * (format == DXGI_FORMAT_R16_UINT ? 2 : 4))),
        m_indexCount(indexCount),
        m_indexFormat(format)
{
    assert(format == DXGI_FORMAT_R16_UINT || format == DXGI_FORMAT_R32_UINT && "IndexBuffer only supports R16_UINT and R32_UINT formats");
    m_indexBufferView = CreateAndGetIndexBufferView();
    SetResourceName(L"Index Buffer");
}
IndexBuffer::IndexBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource, size_t indexCount, DXGI_FORMAT format)
    :   Buffer(device, resource),
        m_indexCount(indexCount),
        m_indexFormat(format)
{
    m_indexBufferView = CreateAndGetIndexBufferView();
    SetResourceName(L"Index Buffer");
}

D3D12_INDEX_BUFFER_VIEW IndexBuffer::CreateAndGetIndexBufferView() const
{
    uint32_t bufferSize = m_indexCount * (m_indexFormat == DXGI_FORMAT_R16_UINT ? 2 : 4);

    D3D12_INDEX_BUFFER_VIEW indexBufferView;
    indexBufferView.BufferLocation = m_resource->GetGPUVirtualAddress();
    indexBufferView.SizeInBytes = bufferSize;
    indexBufferView.Format = m_indexFormat;
    return indexBufferView;
}
