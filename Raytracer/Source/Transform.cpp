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
    
    Matrix m;

    m.Translation(position);
    m += Matrix::CreateFromQuaternion(rotation);
    m += Matrix::CreateScale(scale);

    return m;
}

DirectX::XMFLOAT4X4 Transform::GetMatrix4x4() const
{
    const DirectX::SimpleMath::Matrix mat = GetMatrix();
    DirectX::XMFLOAT4X4 mat4x4;

    DirectX::XMStoreFloat4x4(&mat4x4, mat);
    
    return mat4x4;
}
