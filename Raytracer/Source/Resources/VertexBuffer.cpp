#include "pch.h"
#include "Resources/VertexBuffer.h"

#include "InputElements.h"

VertexBuffer::VertexBuffer(Microsoft::WRL::ComPtr<ID3D12Device5> device, size_t vertexCount, size_t vertexStride)
    :   Buffer(device, CD3DX12_RESOURCE_DESC::Buffer(vertexCount * vertexStride)),
        m_vertexCount(vertexCount),
        m_vertexStride(vertexStride)
{
    m_vertexBufferView = CreateAndGetVertexBufferView();
    SetResourceName(L"Vertex Buffer");
}

VertexBuffer::VertexBuffer(Microsoft::WRL::ComPtr<ID3D12Device5> device, Microsoft::WRL::ComPtr<ID3D12Resource> resource, size_t vertexCount, size_t vertexStride)
    :   Buffer(device, resource),
        m_vertexCount(vertexCount),
        m_vertexStride(vertexStride)
{
    m_vertexBufferView = CreateAndGetVertexBufferView();
    SetResourceName(L"Vertex Buffer");
}

D3D12_VERTEX_BUFFER_VIEW VertexBuffer::CreateAndGetVertexBufferView() const
{
    D3D12_VERTEX_BUFFER_VIEW vbv;
    vbv.BufferLocation = m_resource->GetGPUVirtualAddress();
    vbv.SizeInBytes = static_cast<UINT>(m_vertexStride * m_vertexCount);
    vbv.StrideInBytes = static_cast<UINT>(m_vertexStride);

    return vbv;
}
