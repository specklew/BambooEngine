#pragma once

class ConstantBuffer;
class Model;

class GameObject
{
public:
    GameObject() = default;
    GameObject(const std::shared_ptr<Model>& model, const std::shared_ptr<ConstantBuffer>& modelWorldMatrixBuffer);

    [[nodiscard]] std::shared_ptr<Model> GetModel() const { return m_model; }
    [[nodiscard]] std::shared_ptr<ConstantBuffer> GetWorldMatrixBuffer() const { return m_worldMatrixBuffer; }
    [[nodiscard]] DirectX::XMFLOAT4X4 GetWorldFloat4X4() const { return m_worldMatrix; }
    [[nodiscard]] DirectX::XMMATRIX GetWorldMatrix() const { return DirectX::XMLoadFloat4x4(&m_worldMatrix); }

    void UpdateWorldMatrix(const DirectX::XMFLOAT4X4& worldMatrix);
    void UpdateWorldMatrix() const;

private:
    friend class Renderer; // TODO: Maybe move the instantiation to other class?
    friend class SceneBuilder;
    
    std::shared_ptr<Model> m_model;
    
    std::shared_ptr<ConstantBuffer> m_worldMatrixBuffer;
    DirectX::XMFLOAT4X4 m_worldMatrix;
};
