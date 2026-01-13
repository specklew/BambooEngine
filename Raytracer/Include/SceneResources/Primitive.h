#pragma once

#include "pch.h"

#include "Material.h"

class VertexBuffer;
class IndexBuffer;

struct Primitive
{
    // Primitive uses std::move to take ownership of the buffers.
    Primitive (std::shared_ptr<VertexBuffer> vertexBuffer, std::shared_ptr<IndexBuffer> indexBuffer, std::shared_ptr<Material> material = nullptr)
        : m_vertexBuffer(std::move(vertexBuffer)), m_indexBuffer(std::move(indexBuffer))
    {
        assert(m_vertexBuffer != nullptr && "Vertex buffer cannot be null");
        assert(m_indexBuffer != nullptr && "Index buffer cannot be null");

        if (material)
        {
            m_material = std::move(material);
            return;
        }

        m_material = std::make_shared<Material>();
    }
    std::shared_ptr<VertexBuffer> m_vertexBuffer;
    std::shared_ptr<IndexBuffer> m_indexBuffer;
    std::shared_ptr<Material> m_material;

    [[nodiscard]] std::shared_ptr<VertexBuffer> GetVertexBuffer() const { return m_vertexBuffer; }
    [[nodiscard]] std::shared_ptr<IndexBuffer> GetIndexBuffer() const { return m_indexBuffer; }
    [[nodiscard]] std::shared_ptr<Material> GetMaterial() const { return m_material; }
};
