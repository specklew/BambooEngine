#pragma once
#include "Transform.h"

class Model;

class SceneNode
{
public:

    SceneNode(const std::shared_ptr<SceneNode>& parent = nullptr, const Transform& transform = Transform(), const std::shared_ptr<Model>& model = nullptr);
    
    [[nodiscard]] std::shared_ptr<SceneNode> GetParent() const { return m_parent; }
    [[nodiscard]] const std::vector<std::shared_ptr<SceneNode>>& GetChildren() const { return m_children; }
    [[nodiscard]] std::shared_ptr<Model> GetModel() const { return m_model; }
    [[nodiscard]] const Transform& GetTransform() const { return m_transform; }

    void AddChild(const std::shared_ptr<SceneNode>& child);
    void AddModel(const std::shared_ptr<Model> & model);
    
    // Not sure if I should leave these as const references? TODO: Check
    void SetPosition(const DirectX::SimpleMath::Vector3& position);
    void SetRotation(const DirectX::SimpleMath::Quaternion& rotation);
    void SetRotation(const DirectX::SimpleMath::Vector3& rotation);
    void SetScale(const DirectX::SimpleMath::Vector3& scale);

    //TODO: Add move functions
    
private:
    friend class Scene;

    void UpdateModelConstantBuffer() const;
    
    std::shared_ptr<Model> m_model;
    std::shared_ptr<SceneNode> m_parent;
    std::vector<std::shared_ptr<SceneNode>> m_children;
    Transform m_transform;
};
