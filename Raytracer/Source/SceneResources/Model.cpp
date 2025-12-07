#include "pch.h"
#include "SceneResources/Model.h"

void Model::AddMesh(const std::shared_ptr<Primitive>& mesh)
{
    assert(mesh && "Mesh cannot be null");
    m_meshes.push_back(mesh);
}