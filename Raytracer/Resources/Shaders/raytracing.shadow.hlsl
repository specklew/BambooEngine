#ifndef RAYTRACER_SHADOW_HLSL
#define RAYTRACER_SHADOW_HLSL

#include "RaytracingUtils.hlsl"

struct ShadowPayload
{
    float visibility; // 0 = in shadow, 1 = fully lit
};

float3 GetShadowRayDirection(float3 shadingPoint, LightData light)
{
    if (light.type == 0) // Directional
    {
        return -light.direction; // Light comes from the opposite direction
    }
    if (light.type == 1) // Point
    {
        return normalize(light.position - shadingPoint);
    }
    // Spotlight not implemented
    return float3(0, -1, 0);
}

float GetShadowRayTMax(float3 shadingPoint, LightData light)
{
    if (light.type == 1) // Point — stop at light position to avoid back-side occlusion
        return length(light.position - shadingPoint);
    return RAY_TMAX; // Directional — infinite
}

float TraceShadow(float3 shadingPoint, LightData light)
{
    RayDesc shadowRay;
    shadowRay.Origin = shadingPoint;
    shadowRay.Direction = GetShadowRayDirection(shadingPoint, light);
    shadowRay.TMin = RAY_TMIN;
    shadowRay.TMax = GetShadowRayTMax(shadingPoint, light);

    ShadowPayload payload = { 0.0 }; // Start fully lit
    TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 1, 1, 1, shadowRay, payload);

    return payload.visibility;
}

[shader("anyhit")]
void ShadowHit(inout ShadowPayload payload : SV_RayPayload, Attributes attr)
{
    payload.visibility = 0.0;
    AcceptHitAndEndSearch();
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload : SV_RayPayload)
{
    payload.visibility = 1.0;
}

#endif // RAYTRACER_SHADOW_HLSL