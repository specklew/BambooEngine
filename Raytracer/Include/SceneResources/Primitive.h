#pragma once

#include "pch.h"

#include <cfloat>
#include <vector>

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

    DirectX::XMFLOAT3 m_localAabbMin{  FLT_MAX,  FLT_MAX,  FLT_MAX };
    DirectX::XMFLOAT3 m_localAabbMax{ -FLT_MAX, -FLT_MAX, -FLT_MAX };

    // Local-space copies retained ONLY for emissive primitives: the light-pool
    // bake (SceneBuilder::Build) needs triangle positions after GPU upload.
    std::vector<DirectX::XMFLOAT3> m_emissiveBakePositions;
    std::vector<DirectX::XMFLOAT2> m_emissiveBakeUvs;
    std::vector<uint32_t> m_emissiveBakeIndices;

    [[nodiscard]] std::shared_ptr<Material> GetMaterial() const { return m_material; }
    BufferView GetVertexView() const { return m_vertexBufferOffset; }
    BufferView GetIndexView() const { return m_indexBufferOffset; }
};
