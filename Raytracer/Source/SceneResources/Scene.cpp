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

void SceneBuilder::UpdateMatrices()
{
    UpdateMatricesInNodesRecursively(m_root);
}

void SceneBuilder::PrintDebugInfo()
{
#if _DEBUG
    spdlog::debug("Scene Debug Info:");
    spdlog::debug("Number of models in scene: {}", m_models.size());
    for (int i = 0; i < m_models.size(); i++)
    {
        auto model = m_models[i];
        spdlog::debug("Model {} has {} meshes.", i, model->GetMeshes().size());
        for (int j = 0; j < model->GetMeshes().size(); j++)
        {
            auto mesh = model->GetMeshes()[j];
            spdlog::debug("  Mesh {}: Vertex Buffer Size = {} vertices, Index Buffer Size = {} indices.", 
                         j, 
                         mesh->GetVertexBuffer()->GetVertexCount(), 
                         mesh->GetIndexBuffer()->GetIndexCount());
            
        }
    }

    for (int i = 0; i < m_gameObjects.size(); i++)
    {
        auto gameObject = m_gameObjects[i];
        spdlog::debug("GameObject {} uses Model with {} meshes.", i, gameObject->GetModel()->GetMeshes().size());
        MathUtils::PrintMatrix(gameObject->GetWorldFloat4X4());
    }
#endif
}

std::shared_ptr<Scene> SceneBuilder::Build()
{
    assert(!m_isBuilt && "Scene has already been built");
    m_isBuilt = true;

    PrintDebugInfo();
    UpdateMatricesInNodesRecursively(m_root);
    
    Scene scene;
    scene.m_gameObjects = std::move(m_gameObjects);
    scene.m_models = std::move(m_models);
    scene.m_root = std::move(m_root);
    scene.m_name = std::move(m_name);
    scene.m_rtRepresentation = std::move(m_rtRepresentation);
    
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
