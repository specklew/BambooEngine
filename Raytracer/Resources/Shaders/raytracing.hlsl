#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#include "BRDF.hlsl"
#include "Random.hlsl"
#include "RaytracingUtils.hlsl"
#include "consts.hlsl"
#include "raytracing.shadow.hlsl"
#include "passConstants.hlsl"
#include "RaytraceDebugMode.h"
#include "RaytraceDebugViews.hlsl"

// Minimal hit-ID payload (ADR 0007): the closest hit reports WHAT was hit,
// raygen reconstructs the surface and shades in the bounce loop. The shared
// RaytracingUtils Payload stays for the AO technique.
struct PtPayload
{
    uint   instanceId;
    uint   primitiveId;
    float2 barycentrics;
    uint   hitFlag;  // 1 = ray hit geometry
};

// Sky along a ray. Primary rays (vertex 0) keep the directly-viewed sky at
// full brightness; indirect segments apply the sky-lighting switch + firefly
// clamp. Matches the guided integrator's raygen sky + IndirectSkyRadiance
// pair so both techniques converge to the same target.
float3 SkyRadianceAtVertex(float3 dir, uint vertexIndex)
{
    if (vertexIndex > 0u && skyLightingEnabled == 0)
        return float3(0, 0, 0);
    float u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = -asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    float3 sky = g_skybox.SampleLevel(gsamLinearWrap, float2(u, v), 0).rgb;
    if (vertexIndex > 0u && indirectSkyClamp > 0.0)
        sky = min(sky, indirectSkyClamp.xxx);
    return sky;
}

// ---- Miss ----

// The PT entry points and their TraceRay calls are pipeline-only; a compute
// (inline-RayQuery) compilation of an including file must not see them.
#ifndef GUIDED_TRACE_RQ

[shader("miss")]
void Miss(inout PtPayload payload : SV_RayPayload)
{
    // Sky shading happens in the raygen bounce loop (SkyRadianceAtVertex).
    payload.hitFlag = 0;
}

#endif // GUIDED_TRACE_RQ (shared shading helpers below stay visible)

// ---- Closest hit ----

struct SurfaceData
{
    float3 N;
    float3 V;
    float  NdotV;
    float3 F0;
    float3 albedo;
    float  roughness;
    float  metallic;
};

// ---- Importance-sampled bounce evaluation (PDF-cancelled) ----

// GGX specular bounce. PDF (D * NdotH / (4 * VdotH)) cancels D from the
// microfacet BRDF, leaving: F * G * VdotH / (NdotV * NdotH)
float3 EvalSpecularBounce(SurfaceData s, float3 H, float3 bounceDir)
{
    if (dot(bounceDir, s.N) <= 0.0)
        return float3(0, 0, 0);

    float NdotH = max(dot(s.N, H), EPSILON);
    float NdotL = max(dot(s.N, bounceDir), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);

    float3 F = FresnelSchlick(VdotH, s.F0);
    float G = SmithG_GGX(s.NdotV, NdotL, s.roughness);

    return F * G * VdotH / (s.NdotV * NdotH + EPSILON);
}

// Cosine-weighted diffuse bounce. Lambertian (albedo/PI) divided by
// cosine PDF (NdotL/PI) cancels to kD * albedo.
float3 EvalDiffuseBounce(SurfaceData s, float3 kD, float3 bounceDir)
{
    if (dot(bounceDir, s.N) <= 0.0)
        return float3(0, 0, 0);

    return kD * s.albedo;
}

// ---- Direct lighting BRDF evaluation (no PDF, raw BRDF value) ----

// Cook-Torrance specular for a known light direction.
// Uses direct-lighting geometry term (k = (roughness+1)^2 / 8).
float3 EvalSpecularDirect(SurfaceData s, float3 L)
{
    float3 H = normalize(s.V + L);
    float NdotL = max(dot(s.N, L), EPSILON);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);

    float3 F = FresnelSchlick(VdotH, s.F0);
    float  G = GeometrySmith(s.NdotV, NdotL, s.roughness);
    float  D = DistributionGGX(NdotH, s.roughness);

    return (F * G * D) / (4.0 * s.NdotV * NdotL + EPSILON);
}

// Lambertian diffuse for a known light direction: albedo / PI.
// Energy conservation (kD, metallic gating) is applied by EvalDirectBRDF.
float3 EvalDiffuseDirect(SurfaceData s)
{
    return s.albedo / PI;
}

// Full direct-lighting BRDF (specular + diffuse with Fresnel weighting).
float3 EvalDirectBRDF(SurfaceData s, float3 L)
{
    if (dot(s.N, L) <= 0.0)
        return float3(0, 0, 0);

    float3 F = FresnelSchlick(max(dot(s.V, normalize(s.V + L)), 0.0), s.F0);
    float3 kD = (1.0 - F) * (1.0 - s.metallic);

    return EvalSpecularDirect(s, L) + kD * EvalDiffuseDirect(s);
}

float GetLightAttenuation(float3 shadingPoint, LightData light)
{
    if (light.type == 0) // Directional — no falloff
        return 1.0;

    float dist = length(light.position - shadingPoint);
    float attenuation = 1.0 / (dist * dist + EPSILON);

    // Smooth range cutoff (glTF windowing function)
    if (light.range > 0.0)
    {
        float ratio = dist / light.range;
        float window = saturate(1.0 - ratio * ratio * ratio * ratio);
        attenuation *= window * window;
    }

    return attenuation;
}

float3 CalculateDirectLightning(HitData hit, SurfaceData surface)
{
    float3 directLighting = float3(0, 0, 0);

    for (uint i = 0; i < numLights; i++)
    {
        LightData light = g_lightData[i];
        float3 L = GetShadowRayDirection(hit.position, light);

        float visibility = TraceShadow(hit.position + surface.N * EPSILON, light);
        if (visibility <= 0.0)
            continue;

        float atten = GetLightAttenuation(hit.position, light);
        float3 brdf = EvalDirectBRDF(surface, L);
        directLighting += brdf * light.color * light.intensity * atten * visibility * max(dot(surface.N, L), 0.0);
    }
    return directLighting;
}

// ---- Ray generation ----

#ifndef GUIDED_TRACE_RQ // pipeline-only from here to EOF (TraceRay + entry points)

// Flat iterative path loop (ADR 0007): replaces the recursive closest-hit
// continuation with the same estimator — per vertex add throughput-weighted
// direct light, sample the next bounce, stop after numBounces bounces. All
// rays (primary, bounce, shadow) launch from raygen; the closest hit only
// reports hit IDs and the surface is reconstructed here.
[shader("raygeneration")]
void RayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    uint pixelId = launchIndex.x + launchIndex.y * dims.x;

    float3 accumulated = float3(0, 0, 0);
    for (uint i = 0; i < (uint)samplesPerPixel; i++)
    {
        // Unique seed per pixel, sample, and frame — fully independent across all three
        uint seed = pcg_hash(pixelId ^ (i * 2654435761u) ^ (frameIndex * 805459861u));

        float3 rayOrigin, rayDir;
        GenerateCameraRay(launchIndex, seed, rayOrigin, rayDir);
        seed = pcg_hash(seed);  // advance: camera ray and bounce 0 now use different xi

        float3 radiance = float3(0, 0, 0);
        float3 pathThroughput = float3(1, 1, 1);

        // Vertex 0 = primary hit; direct light at every vertex; a bounce is
        // sampled while vertexIndex < numBounces (numBounces+1 path segments).
        for (uint vertexIndex = 0; vertexIndex <= (uint)numBounces; ++vertexIndex)
        {
            RayDesc ray;
            ray.Origin = rayOrigin;
            ray.Direction = rayDir;
            ray.TMin = RAY_TMIN;
            ray.TMax = RAY_TMAX;

            PtPayload p;
            p.instanceId = 0;
            p.primitiveId = 0;
            p.barycentrics = float2(0, 0);
            p.hitFlag = 0;
            TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, p);

            if (p.hitFlag == 0u)
            {
                radiance += pathThroughput * SkyRadianceAtVertex(rayDir, vertexIndex);
                break;
            }

            InstanceInfo instance = g_instanceInfo[p.instanceId];
            GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
            HitData hit = GetHitData(p.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                                     p.barycentrics, instance.objectToWorld);

            float3 albedo = SampleTextureColor(instance, hit).rgb * instance.baseColorFactor.rgb;
            float2 rm = SampleRoughnessMetallic(instance, hit);
            float roughness = max(rm.x, MIN_ROUGHNESS);
            float metallic = rm.y;

            float3 N = SampleWorldSpaceNormal(instance, hit);
            float3 V = -rayDir;

            // Two-sided shading: flip the shading normal to the side the ray
            // hit. Test with the geometric normal (object-space tri normal ->
            // world), since the normal-mapped N can itself point backward on
            // grazing texels.
            float3 geometricN = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
            if (dot(geometricN, V) < 0.0)
                N = -N;

            // RT debug views; paint and stop if a mode handles this hit.
            RtDebugData rtDebug;
            rtDebug.N = N;
            rtDebug.position = hit.position;
            float3 debugColor;
            if (TryRaytraceDebugView(debugMode, rtDebug, debugColor))
            {
                radiance += pathThroughput * debugColor;
                break;
            }

            SurfaceData surface;
            surface.N         = N;
            surface.V         = V;
            surface.NdotV     = max(dot(N, V), 1e-4);  // div-by-zero guard only; 0.1 floored grazing specular (energy loss)
            surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
            surface.albedo    = albedo;
            surface.roughness = roughness;
            surface.metallic  = metallic;

            radiance += pathThroughput * CalculateDirectLightning(hit, surface);

            if (vertexIndex >= (uint)numBounces)
                break;

            // Continuation sample: stochastic specular/diffuse selection,
            // pdf-cancelled throughput.
            float3 F = FresnelSchlick(surface.NdotV, surface.F0);
            float specularProb = (F.r + F.g + F.b) / 3.0;

            float2 xi = Random2D(seed);
            seed = pcg_hash(seed);  // advance: next bounce gets a different xi

            // Lobe selector must be independent of xi. Reusing a hash of xi
            // conditions the direction sample on the choice -> biased split.
            float pathSelector = Random1D(seed);
            seed = pcg_hash(seed);

            float3 bounceDir;
            float3 throughput;

            if (pathSelector < specularProb)
            {
                float3 H = ImportanceSampleGGX(xi, N, roughness);
                bounceDir = reflect(-V, H);
                throughput = EvalSpecularBounce(surface, H, bounceDir);
                if (all(throughput == 0))
                    break; // invalid bounce sample — direct light at this vertex stands
                throughput /= specularProb;
            }
            else
            {
                float3 kD = (1.0 - F) * (1.0 - metallic);
                bounceDir = CosineSampleHemisphere(xi, N);
                throughput = EvalDiffuseBounce(surface, kD, bounceDir);
                if (all(throughput == 0))
                    break; // invalid bounce sample — direct light at this vertex stands
                throughput /= (1.0 - specularProb);  // branch guarantees pathSelector>=specularProb => 1-specularProb>0
            }

            // BounceHealth: classify a NaN bounce direction at this hit.
            if (debugMode == 2)
            {
                radiance += pathThroughput * BounceHealthColor(bounceDir, N, roughness);
                break;
            }

            pathThroughput *= throughput;
            rayOrigin = hit.position;
            rayDir = bounceDir;
        }

        accumulated += radiance;
    }

    gOutput[launchIndex] = float4(accumulated / samplesPerPixel, 1.0);
}

[shader("anyhit")]
void AnyHit(inout PtPayload payload : SV_RayPayload, in Attributes attr)
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

[shader("closesthit")]
void Hit(inout PtPayload payload : SV_RayPayload, in Attributes attr)
{
    // Report-only (ADR 0007): shading happens in the raygen bounce loop.
    payload.instanceId = InstanceID();
    payload.primitiveId = PrimitiveIndex();
    payload.barycentrics = attr.barycentrics;
    payload.hitFlag = 1;
}

#endif // GUIDED_TRACE_RQ

#endif