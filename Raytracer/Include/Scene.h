#pragma once

class Model;
class SceneNode;

class Scene : public std::enable_shared_from_this<Scene>
{
public:
    Scene();
    void Initialize();
    [[nodiscard]] std::shared_ptr<SceneNode> GetRoot() const { return m_root; }
    [[nodiscard]] const std::vector<std::shared_ptr<Model>>& GetModels() const { return m_models; }
    void PrintDebugInfo();
    void AddFallbackModel(const std::shared_ptr<Model>& model);
    
private:
    friend class SceneNode;
    std::shared_ptr<SceneNode> m_root;
    std::vector<std::shared_ptr<Model>> m_models;
};
