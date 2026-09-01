#ifndef RAYTRACER_SHADOW_HLSL
#define RAYTRACER_SHADOW_HLSL

#include "consts.hlsl"
#include "RaytracingUtils.hlsl"

struct BAMBOO_RAYPAYLOAD ShadowPayload
{
    // write(caller) is NOT redundant, and dropping it produced garbage shadows: with
    // RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH an OPAQUE hit ends traversal without
    // running any shader at all — no any-hit (opaque geometry skips it), no miss — so
    // the shadowed case is the one path where nothing writes the payload. The caller's
    // 0 IS the result there.
    float visibility BAMBOO_PAQ(read(caller) : write(caller, anyhit, miss)); // 0 = in shadow, 1 = fully lit
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
            if (!IsAlphaCutoutTransparent(instance, hit.uv))
                query.CommitNonOpaqueTriangleHit();
        }
    }
    return (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

// Occlusion test along an explicit segment (origin, dir, maxT) instead of a
// LightData — used by pool NEE for sampled emissive-triangle points. TMax is
// pulled in by 2*RAY_TMIN so the sampled light point itself is never its own
// occluder. Same flags/alpha handling as TraceShadow.
float TraceShadowSegment(float3 origin, float3 direction, float maxT)
{
    RayDesc shadowRay;
    shadowRay.Origin = origin;
    shadowRay.Direction = direction;
    shadowRay.TMin = RAY_TMIN;
    shadowRay.TMax = max(RAY_TMIN, maxT - 2 * RAY_TMIN);

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
            if (!IsAlphaCutoutTransparent(instance, hit.uv))
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

    ShadowPayload payload;
    payload.visibility = 0.0; // stands as the result when an opaque hit ends traversal
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

    if (IsAlphaCutoutTransparent(hit.uv))
    {
        IgnoreHit();
    }

    payload.visibility = 0.0;
    AcceptHitAndEndSearch();
}

// Occlusion test along an explicit segment (origin, dir, maxT) instead of a
// LightData — used by pool NEE for sampled emissive-triangle points. TMax is
// pulled in by 2*RAY_TMIN so the sampled light point itself is never its own
// occluder. Same flags/alpha handling as TraceShadow (ShadowHit/ShadowMiss).
float TraceShadowSegment(float3 origin, float3 direction, float maxT)
{
    RayDesc shadowRay;
    shadowRay.Origin = origin;
    shadowRay.Direction = direction;
    shadowRay.TMin = RAY_TMIN;
    shadowRay.TMax = max(RAY_TMIN, maxT - 2 * RAY_TMIN);

    ShadowPayload payload;
    payload.visibility = 0.0; // stands as the result when an opaque hit ends traversal
    TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 1, 1, 1, shadowRay, payload);

    return payload.visibility;
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload : SV_RayPayload)
{
    payload.visibility = 1.0;
}

#endif // GUIDED_TRACE_RQ

#endif // RAYTRACER_SHADOW_HLSL