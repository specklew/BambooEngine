#include "pch.h"
#include "SceneResources/Scene.h"

#include "SceneResources/GameObject.h"
#include "Model.h"
#include "Resources/IndexBuffer.h"
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

std::shared_ptr<Scene> SceneBuilder::Build()
{
    assert(!m_isBuilt && "Scene has already been built");
    m_isBuilt = true;
    
    UpdateMatricesInNodesRecursively(m_root);
    
    Scene scene;
    scene.m_gameObjects = std::move(m_gameObjects);
    scene.m_models = std::move(m_models);
    scene.m_root = std::move(m_root);
    scene.m_name = std::move(m_name);
    scene.m_rtRepresentation = std::move(m_rtRepresentation);
    scene.m_vertexBuffer = std::move(m_vertexBuffer);
    scene.m_indexBuffer = std::move(m_indexBuffer);
    
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
