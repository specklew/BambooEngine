#include "pch.h"
#include "SceneResources/MeshProcessing.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <spdlog/spdlog.h>

extern "C" {
#include <mikktspace/mikktspace.h>
#include <mikktspace/mikktspace.c>
}

#include "AccelerationStructures.h"
#include "Renderer.h"
#include "SceneResources/Scene.h"
#include "SceneResources/Model.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/GameObject.h"
#include "SceneResources/Material.h"

namespace MeshUtils
{

void ComputeNormals(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    using namespace DirectX;

    for (Vertex& vertex : vertices)
        vertex.Normal = {0, 0, 0};

    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        Vertex& v0 = vertices[indices[i]];
        Vertex& v1 = vertices[indices[i + 1]];
        Vertex& v2 = vertices[indices[i + 2]];

        const XMVECTOR p0 = XMLoadFloat3(&v0.Pos);
        const XMVECTOR faceNormal = XMVector3Cross(
            XMVectorSubtract(XMLoadFloat3(&v1.Pos), p0),
            XMVectorSubtract(XMLoadFloat3(&v2.Pos), p0));

        for (Vertex* vertex : { &v0, &v1, &v2 })
            XMStoreFloat3(&vertex->Normal, XMVectorAdd(XMLoadFloat3(&vertex->Normal), faceNormal));
    }

    for (Vertex& vertex : vertices)
    {
        const XMVECTOR accumulated = XMLoadFloat3(&vertex.Normal);
        if (XMVectorGetX(XMVector3LengthSq(accumulated)) > 0.0f)
            XMStoreFloat3(&vertex.Normal, XMVector3Normalize(accumulated));
        else
            vertex.Normal = {0, 1, 0}; // degenerate/unreferenced vertex
    }
}

void ComputeTangents(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    struct UserData
    {
        std::vector<Vertex>* vertices;
        const std::vector<uint32_t>* indices;
    } userData = { &vertices, &indices };

    SMikkTSpaceInterface iface = {};

    iface.m_getNumFaces = [](const SMikkTSpaceContext* pContext) -> int
    {
        const auto* data = static_cast<UserData*>(pContext->m_pUserData);
        return static_cast<int>(data->indices->size() / 3);
    };

    iface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext*, const int) -> int
    {
        return 3;
    };

    iface.m_getPosition = [](const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert)
    {
        const auto* data = static_cast<UserData*>(pContext->m_pUserData);
        const Vertex& v = (*data->vertices)[(*data->indices)[iFace * 3 + iVert]];
        fvPosOut[0] = v.Pos.x; fvPosOut[1] = v.Pos.y; fvPosOut[2] = v.Pos.z;
    };

    iface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert)
    {
        const auto* data = static_cast<UserData*>(pContext->m_pUserData);
        const Vertex& v = (*data->vertices)[(*data->indices)[iFace * 3 + iVert]];
        fvNormOut[0] = v.Normal.x; fvNormOut[1] = v.Normal.y; fvNormOut[2] = v.Normal.z;
    };

    iface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert)
    {
        const auto* data = static_cast<UserData*>(pContext->m_pUserData);
        const Vertex& v = (*data->vertices)[(*data->indices)[iFace * 3 + iVert]];
        fvTexcOut[0] = v.Tex0.x; fvTexcOut[1] = v.Tex0.y;
    };

    iface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert)
    {
        auto* data = static_cast<UserData*>(pContext->m_pUserData);
        Vertex& v = (*data->vertices)[(*data->indices)[iFace * 3 + iVert]];
        v.Tangent = {fvTangent[0], fvTangent[1], fvTangent[2], fSign};
    };

    SMikkTSpaceContext ctx = { &iface, &userData };
    genTangSpaceDefault(&ctx);
}

void DropDegenerateTriangles(const std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    using namespace DirectX;

    std::vector<uint32_t> kept;
    kept.reserve(indices.size());
    size_t dropped = 0;

    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const XMVECTOR p0 = XMLoadFloat3(&vertices[indices[i]].Pos);
        const XMVECTOR cross = XMVector3Cross(
            XMVectorSubtract(XMLoadFloat3(&vertices[indices[i + 1]].Pos), p0),
            XMVectorSubtract(XMLoadFloat3(&vertices[indices[i + 2]].Pos), p0));

        if (XMVectorGetX(XMVector3LengthSq(cross)) > 1e-20f)
        {
            kept.push_back(indices[i]);
            kept.push_back(indices[i + 1]);
            kept.push_back(indices[i + 2]);
        }
        else
        {
            ++dropped;
        }
    }

    if (dropped > 0)
    {
        spdlog::warn("Dropped {} degenerate (zero-area) triangle(s) during load", dropped);
        indices.swap(kept);
    }
}

void EnforceVertexInvariants(std::vector<Vertex>& vertices)
{
    using namespace DirectX;

    for (Vertex& v : vertices)
    {
        if (!std::isfinite(v.Pos.x) || !std::isfinite(v.Pos.y) || !std::isfinite(v.Pos.z))
            v.Pos = {0, 0, 0};
        if (!std::isfinite(v.Tex0.x) || !std::isfinite(v.Tex0.y))
            v.Tex0 = {0, 0};

        XMVECTOR n = XMLoadFloat3(&v.Normal);
        if (!(XMVectorGetX(XMVector3LengthSq(n)) > 1e-12f) || !std::isfinite(v.Normal.x) ||
            !std::isfinite(v.Normal.y) || !std::isfinite(v.Normal.z))
            n = XMVectorSet(0, 1, 0, 0);
        n = XMVector3Normalize(n);
        XMStoreFloat3(&v.Normal, n);

        float w = (v.Tangent.w == 1.0f || v.Tangent.w == -1.0f) ? v.Tangent.w : 1.0f;
        XMVECTOR t = XMVectorSet(v.Tangent.x, v.Tangent.y, v.Tangent.z, 0.0f);
        const bool finite = std::isfinite(v.Tangent.x) && std::isfinite(v.Tangent.y) && std::isfinite(v.Tangent.z);
        const float lenSq = XMVectorGetX(XMVector3LengthSq(t));
        const float align = (lenSq > 1e-12f) ? std::fabs(XMVectorGetX(XMVector3Dot(XMVector3Normalize(t), n))) : 1.0f;

        if (!finite || lenSq <= 1e-12f || align > 0.9999f)
        {
            const XMVECTOR axis = (std::fabs(v.Normal.y) < 0.99f) ? XMVectorSet(0, 1, 0, 0) : XMVectorSet(1, 0, 0, 0);
            t = XMVector3Normalize(XMVector3Cross(axis, n));
        }
        else
        {
            t = XMVector3Normalize(XMVectorSubtract(t, XMVectorScale(n, XMVectorGetX(XMVector3Dot(t, n)))));
        }

        XMFLOAT3 tf; XMStoreFloat3(&tf, t);
        v.Tangent = {tf.x, tf.y, tf.z, w};
    }
}

std::shared_ptr<AccelerationStructures> BuildAccelerationStructures(const Renderer& renderer, const SceneBuilder& scene)
{
    using Microsoft::WRL::ComPtr;

    std::unordered_map<std::shared_ptr<Primitive>, std::shared_ptr<AccelerationStructureBuffers>> modelBLASes;
    std::shared_ptr<AccelerationStructures> as = std::make_shared<AccelerationStructures>();

    for (auto model : scene.GetModels())
    {
        for (auto prim : model->GetMeshes())
        {
            std::vector<BufferView> vertex_buffers;
            std::vector<BufferView> index_buffers;

            vertex_buffers.emplace_back(prim->GetVertexView());
            index_buffers.emplace_back(prim->GetIndexView());

            AccelerationStructureBuffers bottomLevelBuffers = as->CreateBottomLevelAS(
                renderer.g_device.Get(),
                renderer.GetCommandList(),
                vertex_buffers,
                index_buffers,
                prim->GetMaterial()->m_data.isOpaque);

            modelBLASes.emplace(prim, std::make_shared<AccelerationStructureBuffers>(bottomLevelBuffers));
        }
    }

    for (auto go : scene.GetGameObjects())
    {
        auto model = go->GetModel();
        for (auto prim : model->GetMeshes())
        {
            auto blasIt = modelBLASes.find(prim);
            if (blasIt != modelBLASes.end())
            {
                DirectX::XMMATRIX worldMatrix = DirectX::XMLoadFloat4x4(&go->GetWorldFloat4X4());
                auto instance = std::pair(blasIt->second->p_result, worldMatrix);
                as->GetInstances().emplace_back(instance);
            }
            else
            {
                spdlog::warn("Model for GameObject not found in BLASes map during TLAS setup.");
            }
        }
    }

    as->CreateTopLevelAS(Renderer::g_device.Get(), renderer.GetCommandList().Get(), as->GetInstances(), false);

    return as;
}

} // namespace MeshUtils
