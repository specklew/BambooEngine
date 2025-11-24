#include "pch.h"
#include "SceneNode.h"

#include "Model.h"

void SceneNode::AddChild(const std::shared_ptr<SceneNode>& child)
{
    assert(child && "Child node cannot be null");
    m_children.push_back(child);
}

void SceneNode::AddModel(const std::shared_ptr<Model>& model)
{
    assert(model && "Model node cannot be null");
    m_model = model;
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

SceneNode::SceneNode(const std::shared_ptr<SceneNode>& parent, const Transform& transform, const std::shared_ptr<Model>& model) :
    m_model(model),
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
