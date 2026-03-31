#pragma once

enum LightType : uint32_t
{
    Directional = 0,
    Point = 1,
    Spot = 2
};

struct LightData
{
    LightType type;
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 direction;
    DirectX::XMFLOAT3 color;
    float intensity;
    float range;
    // inner / outer cone angles (later)
};
