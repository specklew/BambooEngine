#pragma once

class ConstantBuffer;
struct Primitive;

class Model
{
public:
    Model() = default;
    ~Model() = default;
    std::vector<std::shared_ptr<Primitive>>& GetMeshes() { return m_meshes; }
    void AddMesh(const std::shared_ptr<Primitive>& mesh);
    void UpdateConstantBuffer(DirectX::XMFLOAT4X4 modelWorldMatrix);
private:
    friend class Renderer;

    DirectX::XMFLOAT4X4 m_modelWorldMatrix;
    std::shared_ptr<ConstantBuffer> m_modelWorldMatrixBuffer;
    std::vector<std::shared_ptr<Primitive>> m_meshes;
};
