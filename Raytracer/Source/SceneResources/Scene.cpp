#include "pch.h"
#include "SceneResources/Scene.h"

#include <cfloat>

#include "SceneResources/GameObject.h"
#include "Model.h"
#include "Renderer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/StructuredBuffer.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/Model.h"
#include "SceneResources/Material.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/SceneNode.h"

SceneBuilder::SceneBuilder()
{
    m_root = std::make_shared<SceneNode>();
}

void SceneBuilder::AddGameObject(const std::shared_ptr<GameObject>& gameObject, const std::shared_ptr<Model>& model)
{
    gameObject->m_model = model;
    m_gameObjects.push_back(gameObject);
}

void SceneBuilder::AddModel(const std::shared_ptr<Model>& model)
{
    m_models.push_back(model);
}

void SceneBuilder::AddChild(const std::shared_ptr<SceneNode>& parent, const std::shared_ptr<SceneNode>& child)
{
    parent->AddChild(child);
}

void SceneBuilder::AddLightData(const LightData& lightData)
{
    m_lightData.push_back(lightData);
}

void SceneBuilder::SetName(const std::string& name)
{
    m_name = name;
}

void SceneBuilder::SetAccelerationStructures(const std::shared_ptr<AccelerationStructures>& accelerationStructures)
{
    m_rtRepresentation = accelerationStructures;
}

void SceneBuilder::SetVertexBuffer(const std::shared_ptr<VertexBuffer>& vertexBuffer)
{
    m_vertexBuffer = vertexBuffer;
}

void SceneBuilder::SetIndexBuffer(const std::shared_ptr<IndexBuffer>& indexBuffer)
{
    m_indexBuffer = indexBuffer;
}

void SceneBuilder::UpdateMatrices()
{
    UpdateMatricesInNodesRecursively(m_root);
}

static std::shared_ptr<StructuredBuffer<GeometryInfo>> CreateGeometryInfoBuffer(Renderer& renderer, const std::vector<std::shared_ptr<Primitive>>& primitives)
{
    std::vector<GeometryInfo> models_info;
    
    for (auto primitive : primitives)
    {
        GeometryInfo info = {};
        info.vertexOffset = primitive->GetVertexView().offset;
        info.indexOffset = primitive->GetIndexView().offset;
        models_info.push_back(info);
    }

    auto geo_buffer = renderer.CreateStructuredBuffer(models_info);
    
    return geo_buffer;
}

static std::shared_ptr<StructuredBuffer<InstanceInfo>> CreateInstanceInfoBuffer(Renderer& renderer, const std::vector<std::shared_ptr<GameObject>>& gameObjects, const std::vector<std::shared_ptr<Primitive>>& primitives)
{
    std::vector<InstanceInfo> instances_info;
    
    for (auto go : gameObjects)
    {
        auto model = go->GetModel();

        for (auto primitive : model->GetMeshes())
        {
            auto it = std::find(primitives.begin(), primitives.end(), primitive);
            assert(it != primitives.end() && "Primitive not found in the list of all primitives when creating instance info buffer.");
            int geometry_id = static_cast<int>(std::distance(primitives.begin(), it));
            int texture_id = -1;
            int normal_texture_id = -1;
            int roughness_texture_id = -1;
            float metallic_factor = 1.0f;
            float roughness_factor = 1.0f;
            DirectX::XMFLOAT4 base_color_factor = { 1.0f, 1.0f, 1.0f, 1.0f };

            if (primitive->m_material)
            {
                if (primitive->m_material->m_albedoTexture)
                    texture_id = primitive->m_material->m_albedoTexture->GetTextureIndex();

                if (primitive->m_material->m_normalTexture)
                    normal_texture_id = primitive->m_material->m_normalTexture->GetTextureIndex();

                if (primitive->m_material->m_metallicRoughnessTexture)
                    roughness_texture_id = primitive->m_material->m_metallicRoughnessTexture->GetTextureIndex();

                metallic_factor = primitive->m_material->m_data.metallicFactor;
                roughness_factor = primitive->m_material->m_data.roughnessFactor;
                base_color_factor = primitive->m_material->m_data.baseColorFactor;
            }

            InstanceInfo info = {};
            info.geometryId = geometry_id;
            info.textureId = texture_id;
            info.normalTextureId = normal_texture_id;
            info.roughnessTextureId = roughness_texture_id;
            info.metallicFactor = metallic_factor;
            info.roughnessFactor = roughness_factor;
            info.baseColorFactor = base_color_factor;
            instances_info.push_back(info);
        }
    }

    return renderer.CreateStructuredBuffer(instances_info);
}

static std::shared_ptr<StructuredBuffer<LightData>> CreateLightDataBuffer(Renderer& renderer, const std::vector<LightData>& lightData)
{
    return renderer.CreateStructuredBuffer(lightData);
}

static void ComputeWorldAabb(const std::vector<std::shared_ptr<GameObject>>& gameObjects, DirectX::XMFLOAT3& outMin, DirectX::XMFLOAT3& outMax)
{
    using namespace DirectX;
    XMVECTOR vmin = XMVectorReplicate( FLT_MAX);
    XMVECTOR vmax = XMVectorReplicate(-FLT_MAX);
    bool any_seen = false;

    for (const auto& go : gameObjects)
    {
        const auto model = go->GetModel();
        if (!model) continue;

        const XMMATRIX world = go->GetWorldMatrix();

        for (const auto& primitive : model->GetMeshes())
        {
            const XMFLOAT3 lmin = primitive->m_localAabbMin;
            const XMFLOAT3 lmax = primitive->m_localAabbMax;
            if (lmin.x > lmax.x) continue; // empty primitive

            const XMFLOAT3 corners[8] = {
                { lmin.x, lmin.y, lmin.z }, { lmax.x, lmin.y, lmin.z },
                { lmin.x, lmax.y, lmin.z }, { lmax.x, lmax.y, lmin.z },
                { lmin.x, lmin.y, lmax.z }, { lmax.x, lmin.y, lmax.z },
                { lmin.x, lmax.y, lmax.z }, { lmax.x, lmax.y, lmax.z },
            };
            for (const XMFLOAT3& c : corners)
            {
                XMVECTOR p = XMVector3Transform(XMLoadFloat3(&c), world);
                vmin = XMVectorMin(vmin, p);
                vmax = XMVectorMax(vmax, p);
                any_seen = true;
            }
        }
    }

    if (!any_seen)
    {
        outMin = { -1.0f, -1.0f, -1.0f };
        outMax = {  1.0f,  1.0f,  1.0f };
        return;
    }

    XMStoreFloat3(&outMin, vmin);
    XMStoreFloat3(&outMax, vmax);
}

std::shared_ptr<Scene> SceneBuilder::Build(Renderer& renderer)
{
    assert(!m_isBuilt && "Scene has already been built");
    m_isBuilt = true;
    
    UpdateMatricesInNodesRecursively(m_root);

    auto all_prims = GetAllPrimitives();
    
    auto geo_info_buffer = CreateGeometryInfoBuffer(renderer, all_prims);
    auto instance_info_buffer = CreateInstanceInfoBuffer(renderer, m_gameObjects, all_prims);
    auto light_data_buffer = CreateLightDataBuffer(renderer, m_lightData);

    Scene scene;
    scene.m_lightDataCPU = m_lightData;
    ComputeWorldAabb(m_gameObjects, scene.m_aabbMin, scene.m_aabbMax);
    scene.m_gameObjects = std::move(m_gameObjects);
    scene.m_models = std::move(m_models);
    scene.m_root = std::move(m_root);
    scene.m_name = std::move(m_name);
    scene.m_rtRepresentation = std::move(m_rtRepresentation);
    scene.m_vertexBuffer = std::move(m_vertexBuffer);
    scene.m_indexBuffer = std::move(m_indexBuffer);
    scene.m_geometryInfoBuffer = std::move(geo_info_buffer);
    scene.m_instanceInfoBuffer = std::move(instance_info_buffer);
    scene.m_lightDataBuffer = std::move(light_data_buffer);

    spdlog::info("Scene AABB: min=({:.3f},{:.3f},{:.3f}) max=({:.3f},{:.3f},{:.3f})",
        scene.m_aabbMin.x, scene.m_aabbMin.y, scene.m_aabbMin.z,
        scene.m_aabbMax.x, scene.m_aabbMax.y, scene.m_aabbMax.z);

    return std::make_shared<Scene>(std::move(scene));
}

void SceneBuilder::UpdateMatricesInNodesRecursively(const std::shared_ptr<SceneNode>& node)
{
    node->UpdateModelConstantBuffer();
    for (const auto& child : node->GetChildren())
    {
        UpdateMatricesInNodesRecursively(child);
    }
}

std::vector<std::shared_ptr<Primitive>> SceneBuilder::GetAllPrimitives() const
{
    std::vector<std::shared_ptr<Primitive>> primitives;

    for (auto model : m_models)
    {
        for (auto primitive : model->GetMeshes())
        {
            primitives.emplace_back(primitive);
        }
    }

    return primitives;
}
