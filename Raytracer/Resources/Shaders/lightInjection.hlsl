#ifndef LIGHT_INJECTION_HLSL
#define LIGHT_INJECTION_HLSL

// Reuses scene bindings, BRDF helpers, surface setup and shadow tracing
// from the main path tracing shader.
#include "raytracing.hlsl"
#include "Octahedral.hlsl"

// ---- VXPG light injection resources ----

RWTexture3D<uint> gVoxelIrradiance : register(u1);
RWTexture3D<uint> gVoxelVplCount   : register(u2);

// ShadingPoints G-buffer: primary-hit worldPos in .xyz, octahedral-packed
// normal (bit-cast) in .w. Consumed by superpixel clustering (Stage B).
RWTexture2D<float4> gShadingPoints : register(u3);

// Sentinel for pixels whose primary ray missed: far position, zero-packed
// normal. Clustering treats these as invalid (normal gate fails).
static const float4 SHADINGPOINT_INVALID = float4(1e30, 1e30, 1e30, 0.0);

cbuffer VoxelGridCB : register(b4)
{
    float3 voxGridMin;
    float  voxVoxelSize;
    float3 voxGridMax;
    uint   voxGridDim;
    uint   voxInjectUseAvg;
    uint   _voxReserved0;
    float  voxHeatScale;
    uint   _voxPad0;
}

// Fixed-point irradiance packing (matches SIByL VXPG: scalar = 100)
uint PackIrradiance(float unpacked)
{
    return uint(unpacked * 100.0f);
}

// ---- Injection payload ----

struct InjectPayload
{
    float3 hitPosition;
    float3 result;  // bounce 0: sampled BSDF direction; bounce 1: direct light RGB
    uint   flags;   // 1 = valid hit
    uint   seed;
    uint   bounce;
};

// ---- Miss ----

[shader("miss")]
void InjectMiss(inout InjectPayload payload : SV_RayPayload)
{
    // Only the primary ray (bounce 0) drives the ShadingPoints G-buffer.
    if (payload.bounce == 0)
        gShadingPoints[DispatchRaysIndex().xy] = SHADINGPOINT_INVALID;
    payload.flags = 0;
}

// ---- Any hit (alpha cutout) ----

[shader("anyhit")]
void InjectAnyHit(inout InjectPayload payload : SV_RayPayload, in Attributes attr)
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
}

// ---- Closest hit ----

[shader("closesthit")]
void InjectHit(inout InjectPayload payload : SV_RayPayload, in Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    float3 albedo = SampleTextureColor(hit).rgb * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(hit, instance.roughnessFactor, instance.metallicFactor);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;

    float3 N = SampleWorldSpaceNormal(hit);
    float3 V = -WorldRayDirection();
    float NdotV = max(dot(N, V), 0.1);

    SurfaceData surface;
    surface.N         = N;
    surface.V         = V;
    surface.NdotV     = NdotV;
    surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
    surface.albedo    = albedo;
    surface.roughness = roughness;
    surface.metallic  = metallic;

    payload.hitPosition = hit.position;

    if (payload.bounce == 0)
    {
        // Persist the primary shading point for superpixel clustering. Written
        // here (closest hit) where world position + normal are both available;
        // valid regardless of whether the VPL bounce below succeeds.
        gShadingPoints[DispatchRaysIndex().xy] =
            float4(hit.position, asfloat(UnitVectorToUnorm32Octahedron(N)));

        // Sample one BSDF direction for the VPL ray (same stochastic
        // specular/diffuse selection as the path tracer)
        float3 F = FresnelSchlick(NdotV, surface.F0);
        float specularProb = (F.r + F.g + F.b) / 3.0;

        float2 xi = Random2D(payload.seed);
        payload.seed = pcg_hash(payload.seed);
        float pathSelector = frac(xi.x * 7.13 + xi.y * 3.97);

        float3 bounceDir;
        if (pathSelector < specularProb)
        {
            float3 H = ImportanceSampleGGX(xi, N, roughness);
            bounceDir = reflect(-V, H);
        }
        else
        {
            bounceDir = CosineSampleHemisphere(xi, N);
        }

        if (dot(bounceDir, N) <= 0.0)
        {
            payload.flags = 0;
            return;
        }

        payload.result = bounceDir;
        payload.flags = 1;
    }
    else
    {
        // Second path vertex — evaluate direct light here, raygen injects it
        payload.result = CalculateDirectLightning(hit, surface);
        payload.flags = 1;
    }
}

// ---- Ray generation ----

[shader("raygeneration")]
void InjectRayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    uint pixelId = launchIndex.x + launchIndex.y * dims.x;

    uint seed = pcg_hash(pixelId ^ (frameIndex * 805459861u) ^ 0x9E3779B9u);

    float3 origin, direction;
    GenerateCameraRay(launchIndex, seed, origin, direction);
    seed = pcg_hash(seed);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    // Primary hit
    InjectPayload payload;
    payload.hitPosition = float3(0, 0, 0);
    payload.result = float3(0, 0, 0);
    payload.flags = 0;
    payload.seed = seed;
    payload.bounce = 0;
    TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, payload);
    if (payload.flags == 0)
        return;

    // VPL ray — one BSDF bounce from the primary hit
    RayDesc bounceRay;
    bounceRay.Origin = payload.hitPosition;
    bounceRay.Direction = payload.result;
    bounceRay.TMin = RAY_TMIN;
    bounceRay.TMax = RAY_TMAX;

    payload.flags = 0;
    payload.bounce = 1;
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, bounceRay, payload);
    if (payload.flags == 0)
        return;

    float irradiance = max(payload.result.r, max(payload.result.g, payload.result.b));
    if (irradiance <= 0.0)
        return;

    int3 voxelIdx = int3(floor((payload.hitPosition - voxGridMin) / voxVoxelSize));
    if (any(voxelIdx < 0) || any(voxelIdx >= int(voxGridDim)))
        return;

    uint packedIrr = PackIrradiance(irradiance);
    if (voxInjectUseAvg != 0)
    {
        uint old;
        InterlockedAdd(gVoxelIrradiance[voxelIdx], packedIrr, old);
        InterlockedAdd(gVoxelVplCount[voxelIdx], 1u, old);
    }
    else
    {
        uint old;
        InterlockedMax(gVoxelIrradiance[voxelIdx], packedIrr, old);
        gVoxelVplCount[voxelIdx] = 1u;
    }
}

#endif // LIGHT_INJECTION_HLSL
