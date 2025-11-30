#include "pch.h"
#include "SceneNode.h"

#include "Model.h"
#include "Scene.h"

void SceneNode::AddChild(const std::shared_ptr<SceneNode>& child)
{
    assert(child && "Child node cannot be null");
    assert(m_scene && "SceneNode must belong to a scene before adding children");
    child->m_scene = m_scene;
    child->m_parent = m_parent;
    m_children.push_back(child);
}

void SceneNode::AddModel(const std::shared_ptr<Model>& model)
{
    assert(model && "Model node cannot be null");
    m_model = model;
    m_scene->m_models.emplace_back(m_model);
}

void SceneNode::SetPosition(const DirectX::SimpleMath::Vector3& position)
{
    m_transform.position = position;
    UpdateModelConstantBuffer();
}

void SceneNode::SetRotation(const DirectX::SimpleMath::Quaternion& rotation)
{
    m_transform.rotation = rotation;
    UpdateModelConstantBuffer();
}

void SceneNode::SetRotation(const DirectX::SimpleMath::Vector3& rotation)
{
    m_transform.rotation = DirectX::SimpleMath::Quaternion::CreateFromYawPitchRoll(rotation.y, rotation.x, rotation.z);
    UpdateModelConstantBuffer();
}

void SceneNode::SetScale(const DirectX::SimpleMath::Vector3& scale)
{
    m_transform.scale = scale;
    UpdateModelConstantBuffer();
}

SceneNode::SceneNode(const std::shared_ptr<SceneNode>& parent, const Transform& transform) :
    m_parent(parent),
    m_transform(transform)
{
    UpdateModelConstantBuffer();
}

void SceneNode::UpdateModelConstantBuffer() const
{
    if (!m_model) return;
    m_model->UpdateConstantBuffer(m_transform.GetMatrix4x4());
}
