#pragma once
#include <SimpleMath.h>

class Camera
{
public:
    Camera() = default;
    Camera(DirectX::XMFLOAT3 position, DirectX::XMFLOAT4 rotation);

    void SetPosition(DirectX::XMFLOAT3 position);
    void SetRotation(DirectX::XMFLOAT4 rotation);
    
    void AddPosition(DirectX::XMFLOAT3 position);
    void AddPosition(float x, float y, float z);
    
    void AddRotation(DirectX::XMFLOAT4 rotation);
    void AddRotationEuler(DirectX::XMFLOAT3 rotation);
    void AddRotationEuler(float pitch, float yaw, float roll);
    
    void UpdateMatrices();

    [[nodiscard]] const DirectX::SimpleMath::Vector3& GetPosition() const { return m_position; }
    [[nodiscard]] const DirectX::XMFLOAT4& GetRotation() const { return m_rotation; }
    [[nodiscard]] const DirectX::XMFLOAT4X4& GetViewMatrix() const { return m_viewMatrix; }
    [[nodiscard]] const DirectX::XMFLOAT4X4& GetProjectionMatrix() const { return m_projectionMatrix; }
    [[nodiscard]] const DirectX::XMFLOAT4X4& GetViewProjectionMatrix() const { return m_viewProjectionMatrix; }

    [[nodiscard]] DirectX::SimpleMath::Vector3 GetForward() const { return m_forward; }
    [[nodiscard]] DirectX::SimpleMath::Vector3 GetRight() const { return m_right; }
    [[nodiscard]] DirectX::SimpleMath::Vector3 GetUp() const { return m_up; }

    [[nodiscard]] float GetSpeed() const { return m_speed; }
    
private:
    DirectX::SimpleMath::Vector3 m_position = {0.0f, 0.0f, -10.0f};
    DirectX::SimpleMath::Vector3 m_forward = { 0.0f, 0.0f, 1.0f };
    DirectX::SimpleMath::Vector3 m_right = { 1.0f, 0.0f, 0.0f };
    DirectX::SimpleMath::Vector3 m_up = { 0.0f, 1.0f, 0.0f };
    DirectX::SimpleMath::Quaternion m_rotation = DirectX::SimpleMath::Quaternion::Identity;

    float m_fovYRadians = (DirectX::XM_PI / 180) * 60;
    float m_aspectRatio = 16.0f / 9.0f;
    float m_nearZ = 0.1f;
    float m_farZ = 1000.0f;

    float m_speed = 10.0f;
    
    DirectX::XMFLOAT4X4 m_viewMatrix = {};
    DirectX::XMFLOAT4X4 m_projectionMatrix = {};
    DirectX::XMFLOAT4X4 m_viewProjectionMatrix = {};
};
