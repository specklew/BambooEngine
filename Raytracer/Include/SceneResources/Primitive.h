#pragma once

#include "pch.h"

#include "Material.h"
#include "Resources/BufferView.h"

class VertexBuffer;
class IndexBuffer;

struct Primitive
{
    Primitive (const BufferView& vertexView, const BufferView& indexView, std::shared_ptr<Material> material = nullptr)
    {
        m_vertexBufferOffset = vertexView;
        m_indexBufferOffset = indexView;

        if (material)
        {
            m_material = std::move(material);
            return;
        }

        m_material = std::make_shared<Material>();
    }
    std::shared_ptr<Material> m_material;

    BufferView m_indexBufferOffset;
    BufferView m_vertexBufferOffset;
    
    [[nodiscard]] std::shared_ptr<Material> GetMaterial() const { return m_material; }
    BufferView GetVertexView() const { return m_vertexBufferOffset; }
    BufferView GetIndexView() const { return m_indexBufferOffset; }
};
