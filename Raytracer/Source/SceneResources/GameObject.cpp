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

    m_worldMatrixBuffer->MapDataToWholeBuffer(reinterpret_cast<void**>(&mapped_data));

    DirectX::XMMATRIX W = DirectX::XMLoadFloat4x4(&m_worldMatrix);

    // Transpose for HLSL column-major cbuffer layout (row-vector mul convention)
    DirectX::XMStoreFloat4x4(&mapped_data[0], DirectX::XMMatrixTranspose(W));

    // Upload inverse WITHOUT transpose: HLSL reads it as transpose(W^-1) = (W^-1)^T,
    // which is exactly the inverse-transpose needed for correct normal transforms.
    DirectX::XMVECTOR det;
    DirectX::XMStoreFloat4x4(&mapped_data[1], DirectX::XMMatrixInverse(&det, W));

    m_worldMatrixBuffer->Unmap();
}
