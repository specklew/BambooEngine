#ifndef RAYTRACER_SHADOW_HLSL
#define RAYTRACER_SHADOW_HLSL

#include "consts.hlsl"
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

#ifdef GUIDED_TRACE_RQ

// Inline-RayQuery backend (compute integrator, ADR 0011): same estimator as
// the pipeline path — ACCEPT_FIRST_HIT occlusion with the ShadowHit anyhit's
// alpha test replayed in the candidate loop.
float TraceShadow(float3 shadingPoint, LightData light)
{
    RayDesc shadowRay;
    shadowRay.Origin = shadingPoint;
    shadowRay.Direction = GetShadowRayDirection(shadingPoint, light);
    shadowRay.TMin = RAY_TMIN;
    shadowRay.TMax = GetShadowRayTMax(shadingPoint, light);

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(SceneBVH, RAY_FLAG_NONE, ~0, shadowRay);
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            InstanceInfo instance = g_instanceInfo[query.CandidateInstanceID()];
            uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
            uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
            HitData hit = GetHitData(query.CandidatePrimitiveIndex(), vertexOffset, indexOffset,
                                     query.CandidateTriangleBarycentrics(), instance.objectToWorld);
            float4 albedo = SampleTextureColor(instance, hit) * instance.baseColorFactor;
            if (albedo.a >= EPSILON)
                query.CommitNonOpaqueTriangleHit();
        }
    }
    return (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

#else // pipeline backend

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
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    float4 albedo = SampleTextureColor(hit) * instance.baseColorFactor;
    if (albedo.a < EPSILON)
    {
        IgnoreHit();
    }

    payload.visibility = 0.0;
    AcceptHitAndEndSearch();
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload : SV_RayPayload)
{
    payload.visibility = 1.0;
}

#endif // GUIDED_TRACE_RQ

#endif // RAYTRACER_SHADOW_HLSL