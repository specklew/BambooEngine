#include "pch.h"
#include "SceneResources/GameObject.h"

#include "Utils/Utils.h"
#include "Resources/ConstantBuffer.h"

GameObject::GameObject(const std::shared_ptr<Model>& model, const std::shared_ptr<ConstantBuffer>& modelWorldMatrixBuffer)
    :   m_model(model),
        m_worldMatrixBuffer(modelWorldMatrixBuffer),
        m_worldMatrix(MathUtils::XMFloat4x4Identity())
{
    UpdateWorldMatrix();   
}

void GameObject::UpdateWorldMatrix(const DirectX::XMFLOAT4X4& worldMatrix)
{
    m_worldMatrix = worldMatrix;
    UpdateWorldMatrix();
}

void GameObject::UpdateWorldMatrix() const
{
    auto data_bucket = DirectX::XMFLOAT4X4{};
    DirectX::XMFLOAT4X4* mapped_data = &data_bucket;
    
    m_worldMatrixBuffer->MapDataToWholeBuffer(&mapped_data);
    memcpy(&mapped_data[0], &m_worldMatrix, sizeof(DirectX::XMFLOAT4X4));
    m_worldMatrixBuffer->Unmap();
}
