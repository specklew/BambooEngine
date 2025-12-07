#pragma once

#include "pch.h"

class VertexBuffer;
class IndexBuffer;

struct Primitive
{
    // Primitive uses std::move to take ownership of the buffers.
    Primitive (std::shared_ptr<VertexBuffer> vertexBuffer, std::shared_ptr<IndexBuffer> indexBuffer)
        : m_vertexBuffer(std::move(vertexBuffer)), m_indexBuffer(std::move(indexBuffer))
    {
        assert(m_vertexBuffer != nullptr && "Vertex buffer cannot be null");
        assert(m_indexBuffer != nullptr && "Index buffer cannot be null");
    }

    std::shared_ptr<VertexBuffer> m_vertexBuffer;
    std::shared_ptr<IndexBuffer> m_indexBuffer;

    [[nodiscard]] std::shared_ptr<VertexBuffer> GetVertexBuffer() const { return m_vertexBuffer; }
    [[nodiscard]] std::shared_ptr<IndexBuffer> GetIndexBuffer() const { return m_indexBuffer; }
};
