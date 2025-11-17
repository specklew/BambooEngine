#pragma once

struct Primitive;

class Model
{
public:
    std::vector<std::shared_ptr<Primitive>>& GetMeshes() { return m_meshes; }
private:
    friend class ModelLoading;
    
    Model() = default;

    void AddMesh(const std::shared_ptr<Primitive>& mesh)
    {
        assert(mesh && "Mesh cannot be null");
        m_meshes.push_back(mesh);
    }
    
    std::vector<std::shared_ptr<Primitive>> m_meshes;
};
