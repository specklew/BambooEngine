#include "pch.h"
#include "SceneNode.h"

#include "GameObject.h"

void SceneNode::AddChild(const std::shared_ptr<SceneNode>& child)
{
    assert(child && "Child node cannot be null");
    child->m_parent = shared_from_this();
    m_children.push_back(child);
}

void SceneNode::AddGameObject(const std::shared_ptr<GameObject> & gameObject)
{
    assert(gameObject && "Model node cannot be null");
    m_gameObject = gameObject;
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
    if (!m_gameObject) return;

    const DirectX::XMFLOAT4X4 model_matrix = TraverseParentMatrices();
    m_gameObject->UpdateWorldMatrix(model_matrix);
}

DirectX::SimpleMath::Matrix SceneNode::TraverseParentMatrices() const
{
    if (m_parent)
    {
        return m_transform.GetMatrix() * m_parent->TraverseParentMatrices();
    }
    return m_transform.GetMatrix();
}
