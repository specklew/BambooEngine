#pragma once
#include "Buffer.h"

class VertexBuffer : public Buffer
{
public:
    size_t GetVertexCount() const { return m_vertexCount; }
    size_t GetVertexStride() const { return m_vertexStride; }
    D3D12_VERTEX_BUFFER_VIEW GetVertexBufferView() const { return m_vertexBufferView; }
    
    VertexBuffer(Microsoft::WRL::ComPtr<ID3D12Device5> device, size_t vertexCount, size_t vertexStride);
    VertexBuffer(Microsoft::WRL::ComPtr<ID3D12Device5> device, Microsoft::WRL::ComPtr<ID3D12Resource> resource, size_t vertexCount, size_t vertexStride);

private:
    D3D12_VERTEX_BUFFER_VIEW CreateAndGetVertexBufferView() const;

    size_t m_vertexCount;
    size_t m_vertexStride;
    D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
    
};
