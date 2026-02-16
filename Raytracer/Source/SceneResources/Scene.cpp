#include "pch.h"
#include "SceneResources/Scene.h"

#include "SceneResources/GameObject.h"
#include "Model.h"
#include "Renderer.h"
#include "Resources/ConstantBuffer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/StructuredBuffer.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/Model.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/SceneNode.h"
#include "Utils/Utils.h"

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
            int geometryId = static_cast<int>(std::distance(primitives.begin(), it));
            
            InstanceInfo info = {};
            info.geometryId = geometryId;
            instances_info.push_back(info);
        }
    }

    return renderer.CreateStructuredBuffer(instances_info);
}

std::shared_ptr<Scene> SceneBuilder::Build(Renderer& renderer)
{
    assert(!m_isBuilt && "Scene has already been built");
    m_isBuilt = true;
    
    UpdateMatricesInNodesRecursively(m_root);

    auto all_prims = GetAllPrimitives();
    
    auto geo_info_buffer = CreateGeometryInfoBuffer(renderer, all_prims);
    auto instance_info_buffer = CreateInstanceInfoBuffer(renderer, m_gameObjects, all_prims);
    
    Scene scene;
    scene.m_gameObjects = std::move(m_gameObjects);
    scene.m_models = std::move(m_models);
    scene.m_root = std::move(m_root);
    scene.m_name = std::move(m_name);
    scene.m_rtRepresentation = std::move(m_rtRepresentation);
    scene.m_vertexBuffer = std::move(m_vertexBuffer);
    scene.m_indexBuffer = std::move(m_indexBuffer);
    scene.m_geometryInfoBuffer = std::move(geo_info_buffer);
    scene.m_instanceInfoBuffer = std::move(instance_info_buffer);
    
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
