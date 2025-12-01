#include "pch.h"
#include "Transform.h"

Transform::Transform(DirectX::SimpleMath::Vector3 position, DirectX::SimpleMath::Quaternion rotation, DirectX::SimpleMath::Vector3 scale) :
    position(position),
    rotation(rotation),
    scale(scale)
{
}

DirectX::SimpleMath::Matrix Transform::GetMatrix() const
{
    using DirectX::SimpleMath::Matrix;
    
    return DirectX::XMMatrixAffineTransformation(
        DirectX::XMLoadFloat3(&scale),
        DirectX::g_XMZero,
        DirectX::XMLoadFloat4(&rotation),
        DirectX::XMLoadFloat3(&position));
}

DirectX::XMFLOAT4X4 Transform::GetMatrix4x4() const
{
    const DirectX::SimpleMath::Matrix mat = GetMatrix();
    DirectX::XMFLOAT4X4 mat4x4;

    DirectX::XMStoreFloat4x4(&mat4x4, mat);
    
    return mat4x4;
}
