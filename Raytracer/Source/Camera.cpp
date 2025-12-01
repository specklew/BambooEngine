#include "pch.h"
#include "Camera.h"

using namespace DirectX::SimpleMath;

Camera::Camera(DirectX::XMFLOAT3 position, DirectX::XMFLOAT4 rotation) :
    m_position(position),
    m_rotation(rotation)
{
    UpdateMatrices();
}

void Camera::SetPosition(DirectX::XMFLOAT3 position)
{
    m_position = position;
    UpdateMatrices();
}

void Camera::SetRotation(DirectX::XMFLOAT4 rotation)
{
    m_rotation = rotation;

    // TODO: remove roll with quaternions properly
    Vector3 euler = m_rotation.ToEuler();
    euler.z = 0;
    m_rotation = Quaternion::CreateFromYawPitchRoll(euler.y, euler.x, euler.z);
    
    UpdateMatrices();
}

void Camera::AddPosition(DirectX::XMFLOAT3 position)
{
    SetPosition(position + m_position);
}

void Camera::AddPosition(float x, float y, float z)
{
    AddPosition(DirectX::XMFLOAT3(x, y, z));
}

void Camera::AddRotation(DirectX::XMFLOAT4 rotation)
{
    SetRotation(Quaternion::Concatenate(m_rotation, rotation));
}

void Camera::AddRotationEuler(DirectX::XMFLOAT3 rotation)
{
    AddRotation(Quaternion::CreateFromYawPitchRoll(rotation.y, rotation.x, rotation.z));
}

void Camera::AddRotationEuler(float pitch, float yaw, float roll)
{
    AddRotationEuler(DirectX::XMFLOAT3(pitch, yaw, roll));
}

void Camera::UpdateMatrices()
{
    m_forward = Vector3::Transform(Vector3(0, 0, 1), Quaternion(m_rotation));
    m_right = Vector3::Transform(Vector3(-1, 0, 0), Quaternion(m_rotation));
    m_up = Vector3::Transform(Vector3(0, 1, 0), Quaternion(m_rotation));

    Vector3 target = m_forward + m_position;
    Matrix view = Matrix::CreateLookAt(m_position, target, m_up);
    XMStoreFloat4x4(&m_viewMatrix, view);

    Matrix proj = Matrix::CreatePerspectiveFieldOfView(m_fovYRadians, m_aspectRatio, m_nearZ, m_farZ);
    XMStoreFloat4x4(&m_projectionMatrix, proj);

    Matrix viewProj = view * proj;
    XMStoreFloat4x4(&m_viewProjectionMatrix, viewProj);
}
