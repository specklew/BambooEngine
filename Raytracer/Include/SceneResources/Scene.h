#pragma once

class AssetId;
class Renderer;
struct Primitive;
class GameObject;
class Model;
class SceneNode;
class SceneBuilder;

class Scene
{
public:
    [[nodiscard]] std::shared_ptr<SceneNode> GetRoot() const { return m_root; }
    [[nodiscard]] const std::vector<std::shared_ptr<Model>>& GetModels() const { return m_models; }
    [[nodiscard]] const std::vector<std::shared_ptr<GameObject>>& GetGameObjects() const { return m_gameObjects; }
    
private:
    friend class SceneBuilder;
    Scene() = default;
    
    std::shared_ptr<SceneNode> m_root;
    std::vector<std::shared_ptr<GameObject>> m_gameObjects;
    std::vector<std::shared_ptr<Model>> m_models;

};

class SceneBuilder
{
public:
    SceneBuilder();
    void AddGameObject(const std::shared_ptr<GameObject>& gameObject, const std::shared_ptr<Model>& model);
    void AddModel(const std::shared_ptr<Model>& model);
    void AddChild(const std::shared_ptr<SceneNode>& parent, const std::shared_ptr<SceneNode>& child);

    std::shared_ptr<SceneNode> GetRoot() const { return m_root; }
    std::vector<std::shared_ptr<GameObject>> GetGameObjects() const { return m_gameObjects; }
    std::shared_ptr<Model> GetModel(const int index) const { return m_models[index]; }

    void PrintDebugInfo();
    
    std::shared_ptr<Scene> Build();
private:
    friend class SceneNode;

    static void UpdateMatricesInNodesRecursively(const std::shared_ptr<SceneNode>& node);
    
    std::vector<std::shared_ptr<GameObject>> m_gameObjects;
    std::vector<std::shared_ptr<Model>> m_models;
    std::shared_ptr<SceneNode> m_root;

    bool m_isBuilt = false;
};