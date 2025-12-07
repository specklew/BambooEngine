#pragma once
#include "Transform.h"

class GameObject;
class Model;

class SceneNode : public std::enable_shared_from_this<SceneNode>
{
public:
    explicit SceneNode(const std::shared_ptr<SceneNode>& parent = nullptr, const Transform& transform = Transform());
    
    [[nodiscard]] std::shared_ptr<SceneNode> GetParent() const { return m_parent; }
    [[nodiscard]] const std::vector<std::shared_ptr<SceneNode>>& GetChildren() const { return m_children; }
    [[nodiscard]] std::shared_ptr<GameObject> GetGameObject() const { return m_gameObject; }
    [[nodiscard]] const Transform& GetTransform() const { return m_transform; }

    void AddChild(const std::shared_ptr<SceneNode>& child);
    void AddGameObject(const std::shared_ptr<GameObject> & gameObject);
    
    // Not sure if I should leave these as const references? TODO: Check
    void SetPosition(const DirectX::SimpleMath::Vector3& position);
    void SetRotation(const DirectX::SimpleMath::Quaternion& rotation);
    void SetRotation(const DirectX::SimpleMath::Vector3& rotation);
    void SetScale(const DirectX::SimpleMath::Vector3& scale);

    //TODO: Add move functions

private:
    friend class SceneBuilder;

    void UpdateModelConstantBuffer() const;
    DirectX::SimpleMath::Matrix TraverseParentMatrices() const;
    
    std::shared_ptr<SceneNode> m_parent;
    std::vector<std::shared_ptr<SceneNode>> m_children;
    std::shared_ptr<GameObject> m_gameObject;
    Transform m_transform;
};
