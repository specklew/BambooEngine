#pragma once

struct Material;
class ConstantBuffer;
struct Primitive;

class Model
{
public:
    std::vector<std::shared_ptr<Primitive>>& GetMeshes() { return m_meshes; }
    void AddMesh(const std::shared_ptr<Primitive>& mesh);
private:
    std::vector<std::shared_ptr<Primitive>> m_meshes;
};
