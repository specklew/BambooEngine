#include "BRDF.hlsl"
#include "Random.hlsl"
#include "RaytracingUtils.hlsl"
#include "raytracing.shadow.hlsl"
#include "passConstants.hlsl"

static const float AO_RADIUS = 5.0f;

// Cast one AO probe ray. Reuses ShadowPayload with explicit direction + radius
// instead of LightData. Returns 1.0 if unoccluded, 0.0 if occluded.
float TraceAOProbe(float3 origin, float3 direction, float radius)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = direction;
    ray.TMin      = RAY_TMIN;
    ray.TMax      = radius;

    ShadowPayload payload = { 0.0f };
    TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 1, 1, 1, ray, payload);
    return payload.visibility;
}

// ---- Ray generation ----

[shader("raygeneration")]
void RayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 dims        = DispatchRaysDimensions().xy;
    uint  pixelId     = launchIndex.x + launchIndex.y * dims.x;

    // One primary ray per pixel per frame; frameIndex drives temporal variation
    uint seed = pcg_hash(pixelId ^ (frameIndex * 805459861u));

    float3 origin, direction;
    GenerateCameraRay(launchIndex, seed, origin, direction);
    seed = pcg_hash(seed);

    RayDesc primaryRay;
    primaryRay.Origin    = origin;
    primaryRay.Direction = direction;
    primaryRay.TMin      = RAY_TMIN;
    primaryRay.TMax      = RAY_TMAX;

    // Repurpose Payload: color = hit world position,
    //                   throughput = hit world normal,
    //                   bounceCount = hit flag (0 = miss, 1 = surface)
    Payload payload;
    payload.color       = float3(0, 0, 0);
    payload.throughput  = float3(0, 0, 0);
    payload.bounceCount = 0;
    payload.seed        = seed;

    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, primaryRay, payload);

    if (payload.bounceCount == 0)
    {
        // Primary ray hit sky — fully unoccluded
        gOutput[launchIndex] = float4(1, 1, 1, 1);
        return;
    }

    float3 hitPos    = payload.color;
    float3 hitNormal = payload.throughput;
    uint   probeSeed = payload.seed;

    // Cast samplesPerPixel cosine-weighted probe rays; temporal accumulation
    // blends results across frames for noise convergence.
    float unoccluded = 0.0f;
    for (uint i = 0; i < (uint)samplesPerPixel; i++)
    {
        float2 xi       = Random2D(probeSeed);
        probeSeed       = pcg_hash(probeSeed);
        float3 probeDir = CosineSampleHemisphere(xi, hitNormal);
        float3 probeOrg = hitPos + hitNormal * RAY_TMIN;
        unoccluded += TraceAOProbe(probeOrg, probeDir, AO_RADIUS);
    }

    float ao = unoccluded / (float)samplesPerPixel;
    gOutput[launchIndex] = float4(ao, ao, ao, 1.0f);
}

// ---- Miss ----

[shader("miss")]
void Miss(inout Payload payload : SV_RayPayload)
{
    payload.bounceCount = 0; // no surface hit
}

// ---- Closest hit ----

[shader("closesthit")]
void Hit(inout Payload payload : SV_RayPayload, Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset  = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    payload.color       = hit.position;
    payload.throughput  = SampleWorldSpaceNormal(hit);
    payload.bounceCount = 1; // surface was hit
    payload.seed        = pcg_hash(payload.seed);
}
