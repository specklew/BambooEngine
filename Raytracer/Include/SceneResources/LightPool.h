#pragma once
#include <DirectXMath.h>

#include "LightData.h"

// GPU-mirrored structs (tightly packed, HLSL StructuredBuffer layout).
struct EmissiveTriangle
{
    DirectX::XMFLOAT3 v0;
    DirectX::XMFLOAT3 v1;
    DirectX::XMFLOAT3 v2;
    DirectX::XMFLOAT3 radiance;   // Le, constant over the triangle
    float area;
    uint32_t instanceId;
};

enum LightPoolKind : uint32_t
{
    LightPoolAnalytic = 0,
    LightPoolEmissiveTriangle = 1,
};

struct LightPoolEntry
{
    uint32_t kind;      // LightPoolKind
    uint32_t dataIndex; // index into g_lightData or the emissive-triangle buffer
    float power;        // selection weight (quality-only, never a pdf source of bias)
    float cdf;          // inclusive prefix sum, normalized so the last entry is 1.0
};

// Shared power formulas: used both by the initial pool build (Scene.cpp) and
// the runtime analytic-tail rebuild (Renderer.cpp, on light-data dirty) — kept
// here so the two never drift apart.
inline float LightPoolLuminance(const DirectX::XMFLOAT3& c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

inline float AnalyticLightPoolPower(const LightData& light)
{
    constexpr float kPi = 3.14159265358979323846f;
    const DirectX::XMFLOAT3 c = { light.color.x * light.intensity, light.color.y * light.intensity, light.color.z * light.intensity };
    return LightPoolLuminance(c) * 4.0f * kPi;
}
