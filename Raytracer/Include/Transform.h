#pragma once
#include "SimpleMath.h"

class ConstantBuffer;

struct Transform
{
    explicit Transform(DirectX::SimpleMath::Vector3 position = {0.0f, 0.0f, 0.0f},
                       DirectX::SimpleMath::Quaternion rotation = {0.0f, 0.0f, 0.0f, 1.0f},
                       DirectX::SimpleMath::Vector3 scale = {1.0f, 1.0f, 1.0f});

    DirectX::SimpleMath::Matrix GetMatrix() const;
    DirectX::XMFLOAT4X4 GetMatrix4x4() const;
    
    DirectX::SimpleMath::Vector3 position;
    DirectX::SimpleMath::Quaternion rotation;
    DirectX::SimpleMath::Vector3 scale;
};