#include "pch.h"
#include "Scene.h"

#include "Model.h"
#include "Primitive.h"
#include "SceneNode.h"
#include "Resources/IndexBuffer.h"
#include "Resources/VertexBuffer.h"

Scene::Scene()
{
    m_root = std::make_shared<SceneNode>();
}

void Scene::Initialize()
{
    m_root->m_scene = shared_from_this();
}

void Scene::PrintDebugInfo()
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
#endif
}

void Scene::AddFallbackModel(const std::shared_ptr<Model>& model)
{
    m_models.push_back(model);
}
