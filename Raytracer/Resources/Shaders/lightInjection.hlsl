#ifndef LIGHT_INJECTION_HLSL
#define LIGHT_INJECTION_HLSL

// Reuses scene bindings, BRDF helpers, surface setup and shadow tracing
// from the main path tracing shader.
#include "raytracing.hlsl"
#include "Octahedral.hlsl"
#include "VBuffer.hlsl"

// ---- VXPG light injection resources ----

RWTexture3D<uint> gVoxelIrradiance : register(u1);
RWTexture3D<uint> gVoxelVplCount   : register(u2);

// ShadingPoints G-buffer: primary-hit worldPos in .xyz, octahedral-packed
// normal (bit-cast) in .w. Consumed by superpixel clustering (Stage B).
RWTexture2D<float4> gShadingPoints : register(u3);

// VXPG faithful port (B+): per-voxel representative VPL (second-vertex surface
// pos + octa normal, last-writer-wins) for the fingerprint pass; per-pixel VPL
// hit position for cvis assignment. Written at the bounce-1 closest hit.
RWTexture3D<float4> gVoxelRepresentative : register(u4);
RWTexture2D<float4> gVplPosition         : register(u5);

// Shared primary-visibility buffer (ADR 0004): the primary hit comes from
// here; injection no longer traces its own camera rays.
RWTexture2D<uint4> gVBuffer : register(u6);

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
    uint   voxReuseGiVpl; // ADR 0009: 1 = GI's BSDF subtree writes the VPL data
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

    // Two-sided shading: flip to the ray's side (same as PT / guided PT) so
    // ShadingPoints, the representative VPL and direct light get a usable
    // normal on back-face hits.
    float3 geometricNormal = normalize(mul((float3x3)ObjectToWorld3x4(), hit.tri_normal));
    if (dot(geometricNormal, V) < 0.0)
        N = -N;

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

    // Only the VPL bounce is traced now — the primary hit comes from the
    // VBuffer and is shaded in raygen. This is the second path vertex:
    // evaluate direct light here, raygen injects it.
    payload.result = CalculateDirectLightning(hit, surface);
    payload.flags = 1;

    // VXPG B+: stash the representative VPL (pos + normal) for this voxel and
    // the per-pixel VPL position. N and the launch index are both available
    // here, so no payload round-trip is needed. Last-writer-wins per voxel.
    const float packedN = asfloat(UnitVectorToUnorm32Octahedron(N));
    gVplPosition[DispatchRaysIndex().xy] = float4(hit.position, packedN);

    int3 vxRep = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
    if (all(vxRep >= 0) && all(vxRep < int(voxGridDim)))
        gVoxelRepresentative[vxRep] = float4(hit.position, packedN);
}

// ---- Ray generation ----

[shader("raygeneration")]
void InjectRayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    uint pixelId = launchIndex.x + launchIndex.y * dims.x;

    // Reuse config (ADR 0009): the guided GI's BSDF subtree owns every VPL
    // write (including the per-pixel clear); this pass only emits the
    // ShadingPoints G-buffer below. Faithful config clears + traces here.
    const bool reuseGiVpl = (voxReuseGiVpl != 0u);

    // Clear this pixel's VPL position; the bounce-1 closest hit overwrites it on
    // a hit, so pixels whose bounce misses stay zero (cvis treats zero as empty).
    if (!reuseGiVpl)
        gVplPosition[launchIndex] = float4(0, 0, 0, 0);

    // Primary hit from the shared VBuffer (ADR 0004) — reconstruct instead of
    // tracing. All VBuffer consumers see the exact same hit.
    VBufferData vb = UnpackVBufferData(gVBuffer[launchIndex]);
    if (IsVBufferInvalid(vb))
    {
        gShadingPoints[launchIndex] = SHADINGPOINT_INVALID;
        return;
    }

    InstanceInfo instance = g_instanceInfo[vb.instanceId];
    GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
    HitData hit = GetHitData(vb.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                             vb.barycentrics, instance.objectToWorld);

    float3 N = SampleWorldSpaceNormal(instance, hit);
    float3 V = normalize(cameraWorldPos - hit.position);

    // Two-sided shading: flip to the camera's side (same as the closest hit).
    float3 geometricNormal = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
    if (dot(geometricNormal, V) < 0.0)
        N = -N;

    // Persist the primary shading point for superpixel clustering; valid
    // regardless of whether the VPL bounce below succeeds.
    gShadingPoints[launchIndex] = float4(hit.position, asfloat(UnitVectorToUnorm32Octahedron(N)));

    if (reuseGiVpl)
        return; // VPL trace + injection happen in the guided GI raygen (ADR 0009)

    // Sample one BSDF direction for the VPL ray (same stochastic
    // specular/diffuse selection as the path tracer)
    float3 albedo = SampleTextureColor(instance, hit).rgb * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(instance, hit);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;

    float NdotV = max(dot(N, V), 0.1);
    float3 F0 = lerp(DIELECTRIC_F0, albedo, metallic);
    float3 F = FresnelSchlick(NdotV, F0);
    float specularProb = (F.r + F.g + F.b) / 3.0;

    uint seed = pcg_hash(pixelId ^ (frameIndex * 805459861u) ^ 0x9E3779B9u);
    float2 xi = Random2D(seed);
    seed = pcg_hash(seed);
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
        return;

    // VPL ray — one BSDF bounce from the primary hit
    RayDesc bounceRay;
    bounceRay.Origin = hit.position;
    bounceRay.Direction = bounceDir;
    bounceRay.TMin = RAY_TMIN;
    bounceRay.TMax = RAY_TMAX;

    InjectPayload payload;
    payload.hitPosition = float3(0, 0, 0);
    payload.result = float3(0, 0, 0);
    payload.flags = 0;
    payload.seed = seed;
    payload.bounce = 1;
    TraceRay(SceneBVH, 0, ~0, 0, 1, 0, bounceRay, payload);
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
