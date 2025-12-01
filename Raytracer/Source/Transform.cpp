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

    // Note: The order of transformations is Scale -> Rotate -> Translate
    Matrix m = Matrix::Identity;
    m *= Matrix::CreateScale(scale);
    m *= Matrix::CreateFromQuaternion(rotation);
    m *= Matrix::CreateTranslation(position * 10); // Scale up the position by a factor of 10

    return m;
}

DirectX::XMFLOAT4X4 Transform::GetMatrix4x4() const
{
    const DirectX::SimpleMath::Matrix mat = GetMatrix();
    DirectX::XMFLOAT4X4 mat4x4;

    DirectX::XMStoreFloat4x4(&mat4x4, mat);
    
    return mat4x4;
}
