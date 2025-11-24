#pragma once

class SceneNode;

class Scene
{
public:
    Scene();
    
    [[nodiscard]] std::shared_ptr<SceneNode> GetRoot() const { return m_root; }
    
private:
    std::shared_ptr<SceneNode> m_root;
};
