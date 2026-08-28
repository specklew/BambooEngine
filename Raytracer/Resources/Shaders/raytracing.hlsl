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
#include "PassRegisters.h"

// 1 = debug-view branches compiled in (default; interactive debug variant),
// 0 = clean benchmark variant, compiled through the "noviews" vendor lever
// (VendorLevers::VariantAsset suffixes the raygen asset id; there is no sidecar
// .clean.shader file any more). The CVar-driven
// branches are wave-uniform and cheap at runtime; compiling them out removes
// the dead code's register/I-cache footprint from the hot raygen.
#ifndef RT_DEBUG_VIEWS
#define RT_DEBUG_VIEWS 1
#endif

// 1 = launch index is remapped to a pixel in Morton order within a tile (the
// "swizzle" vendor lever, ADR 0020 R2), so the pixels a wave shades form a
// compact block rather than a wide scanline strip. Vendor-neutral, and the
// control experiment for SER: it separates "reordering helps this workload"
// from "this GPU has a reorder unit".
#ifndef RAYGEN_SWIZZLE
#define RAYGEN_SWIZZLE 0
#endif

#if RAYGEN_SWIZZLE

uint CompactEveryOtherBit(uint x)
{
    x &= 0x55555555u;
    x = (x ^ (x >> 1)) & 0x33333333u;
    x = (x ^ (x >> 2)) & 0x0f0f0f0fu;
    x = (x ^ (x >> 4)) & 0x00ff00ffu;
    x = (x ^ (x >> 8)) & 0x0000ffffu;
    return x;
}

uint2 MortonDecode2D(uint index)
{
    return uint2(CompactEveryOtherBit(index), CompactEveryOtherBit(index >> 1));
}

// Bijective on the PADDED launch grid only: the tile part of the index passes
// through untouched and the Morton decode permutes the slots inside one tile, so
// every pixel is still hit exactly once — but only if the grid covers whole
// tiles. The dispatch is padded up for that (DxrPass::Render), which is also why
// callers must drop the mapped pixels that land outside the image.
uint2 SwizzleLaunchToPixel(uint2 launchIndex)
{
    const uint2 tile  = launchIndex >> RAYGEN_SWIZZLE_TILE_SHIFT;
    const uint2 local = launchIndex & (RAYGEN_SWIZZLE_TILE_SIZE - 1);
    const uint  slot  = (local.y << RAYGEN_SWIZZLE_TILE_SHIFT) | local.x;
    return (tile << RAYGEN_SWIZZLE_TILE_SHIFT) + MortonDecode2D(slot);
}

#endif // RAYGEN_SWIZZLE

// Indirect-only output (guidingFlags bit 12). The VXPG paper evaluates on images that
// "visualize indirect illumination only, omitting direct illumination, to emphasize the
// improvement of our algorithm for guiding indirect illumination" (Sec. 6), so a
// paper-comparable measurement has to drop exactly the same term: everything reaching
// the FIRST vertex without a bounce — its NEE, its own emission, and directly visible
// sky. Everything that arrives via one or more bounces stays, including the direct
// lighting evaluated at deeper vertices, which IS the indirect illumination of the image.
// Applies to PT and to the guided integrator alike, and must be set for the reference
// render too or the comparison scores an indirect image against a full one.
bool IndirectOnly() { return (guidingFlags & (1u << 12)) != 0u; }

// Minimal hit-ID payload (ADR 0007): the closest hit reports WHAT was hit,
// raygen reconstructs the surface and shades in the bounce loop. The shared
// RaytracingUtils Payload stays for the AO technique.
struct BAMBOO_RAYPAYLOAD PtPayload
{
    uint   instanceId   BAMBOO_PAQ(read(caller) : write(closesthit));
    uint   primitiveId  BAMBOO_PAQ(read(caller) : write(closesthit));
    float2 barycentrics BAMBOO_PAQ(read(caller) : write(closesthit));
    // The only field a miss writes, and the only one read on a missed ray.
    uint   hitFlag      BAMBOO_PAQ(read(caller) : write(closesthit, miss));
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

// ---- Shared BSDF-mixture pdf (matches ImportanceSampleGGX/CosineSampleHemisphere) ----
// PT, the light pool, and the guided integrator share this one definition; the
// guided shader's bit-identical PdfBsdf/PdfGGX/PdfCosine were removed in Task 7.

float PdfBsdfMixture(SurfaceData s, float specularProb, float3 dir)
{
    if (dot(dir, s.N) <= 0.0)
        return 0.0;
    float3 H = normalize(s.V + dir);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);
    float pdfSpecular = DistributionGGX(NdotH, s.roughness) * NdotH / (4.0 * VdotH);
    float pdfDiffuse = max(dot(s.N, dir), 0.0) / PI;
    return specularProb * pdfSpecular + (1.0 - specularProb) * pdfDiffuse;
}

float SurfaceSpecularProb(SurfaceData s)
{
    float3 F = FresnelSchlick(s.NdotV, s.F0);
    return (F.r + F.g + F.b) / 3.0;
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

// Full BRDF for a known direction, matching what the BSDF and guide strategies
// effectively evaluate: path-tracing Smith G (k = a^2/2) and kD taken at NdotV,
// i.e. the pdf-cancelled bounce estimator above written out without the cancellation.
// Area-emitter NEE evaluates with THIS, not EvalDirectBRDF: NEE and BSDF/guide
// MIS-mix over the same emitter, and a G/kD convention split between them makes the
// converged specular response a weight-proportional blend of two BRDFs (ADR 0016 M4).
// Delta lights keep EvalDirectBRDF — no strategy overlaps NEE there.
float3 EvalPathBRDF(SurfaceData s, float3 L)
{
    float NdotL = dot(s.N, L);
    if (NdotL <= 0.0)
        return float3(0, 0, 0);
    NdotL = max(NdotL, EPSILON);

    float3 H = normalize(s.V + L);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);

    float  D = DistributionGGX(NdotH, s.roughness);
    float  G = SmithG_GGX(s.NdotV, NdotL, s.roughness);
    float3 F = FresnelSchlick(VdotH, s.F0);
    float3 specular = (D * G * F) / (4.0 * s.NdotV * NdotL + EPSILON);

    float3 Fn = FresnelSchlick(s.NdotV, s.F0);
    float3 kD = (1.0 - Fn) * (1.0 - s.metallic);

    return specular + kD * s.albedo / PI;
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

// CalculateDirectLightning (per-light analytic loop) was removed in Task 7 — its
// last callers (PT and the guided integrator) now use SampleDirectLight from the
// unified pool below. GetLightAttenuation / EvalDirectBRDF / GetShadowRayDirection
// remain: SampleDirectLight's analytic-delta branch reuses them.

#include "LightPool.hlsl"

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
#if RAYGEN_SWIZZLE
    // Padded launch grid: dims are the padded extent, so the image size comes
    // from the output texture and the out-of-image slots return.
    uint imageWidth, imageHeight;
    gOutput.GetDimensions(imageWidth, imageHeight);
    const uint2 dims = uint2(imageWidth, imageHeight);
    const uint2 launchIndex = SwizzleLaunchToPixel(DispatchRaysIndex().xy);
    if (launchIndex.x >= dims.x || launchIndex.y >= dims.y)
        return;
#else
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 dims = DispatchRaysDimensions().xy;
#endif
    uint pixelId = launchIndex.x + launchIndex.y * dims.x;

    float3 accumulated = float3(0, 0, 0);
    for (uint i = 0; i < (uint)samplesPerPixel; i++)
    {
        // Unique seed per pixel, sample, and frame — fully independent across all three
        uint seed = pcg_hash(pixelId ^ (i * 2654435761u) ^ (frameIndex * 805459861u));

        float3 rayOrigin, rayDir;
        GenerateCameraRay(launchIndex, seed, float2(dims), rayOrigin, rayDir);
        seed = pcg_hash(seed);  // advance: camera ray and bounce 0 now use different xi

        float3 radiance = float3(0, 0, 0);
        float3 pathThroughput = float3(1, 1, 1);
        float3 previousVertexPosition = rayOrigin;
        float prevBouncePdf = 0.0; // vertex 0 = camera ray (full weight)

        // Vertex 0 = primary hit; direct light at every vertex; a bounce is
        // sampled while vertexIndex < numBounces (numBounces+1 path segments).
        for (uint vertexIndex = 0; vertexIndex <= (uint)numBounces; ++vertexIndex)
        {
            RayDesc ray;
            ray.Origin = rayOrigin;
            ray.Direction = rayDir;
            ray.TMin = RAY_TMIN;
            ray.TMax = RAY_TMAX;

            // Nothing here needs a caller-side value: the closest hit writes every
            // field and the miss writes hitFlag, which is what gates reading the rest.
            // Under PAQ the zeroing would also force the payload to be carried INTO
            // the trace, which is the cost the qualifiers exist to remove.
            PtPayload p;
#if !PAYLOAD_QUALIFIERS
            p.instanceId = 0;
            p.primitiveId = 0;
            p.barycentrics = float2(0, 0);
            p.hitFlag = 0;
#endif
#if RAYGEN_SER
            dx::HitObject hitObject = dx::HitObject::TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, p);
            dx::MaybeReorderThread(hitObject);
            dx::HitObject::Invoke(hitObject, p);
#else
            TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, p);
#endif

            if (p.hitFlag == 0u)
            {
                if (!(vertexIndex == 0u && IndirectOnly()))
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

#if RT_DEBUG_VIEWS
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
#endif

            SurfaceData surface;
            surface.N         = N;
            surface.V         = V;
            surface.NdotV     = max(dot(N, V), 1e-4);  // div-by-zero guard only; 0.1 floored grazing specular (energy loss)
            surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
            surface.albedo    = albedo;
            surface.roughness = roughness;
            surface.metallic  = metallic;

            // Same reasoning: under indirect-only the first vertex's own emission is
            // dropped, so its NEE-pdf query is dead work too.
            if (instance.emissiveLightOffset >= 0 && any(instance.emissiveRadiance > 0.0) &&
                !(vertexIndex == 0u && IndirectOnly()))
            {
                float3 geometricLightNormal = geometricN;
                if (dot(geometricLightNormal, V) > 0.0) // front face only
                {
                    float weight = 1.0;
                    if (vertexIndex > 0u)
                    {
                        float pdfNee = PdfNeeTowardHit(previousVertexPosition, instance, p.primitiveId, hit.position);
                        weight = BalanceWeight(prevBouncePdf, pdfNee, 0.0);
                    }
                    radiance += pathThroughput * EmitterRadiance(instance, hit.uv) * weight;
                }
            }

            // Skip the CALL, not just the accumulation: SampleDirectLight samples the
            // light pool and traces a shadow ray, and indirect-only used to pay for both
            // and then throw the result away.
            if (!(vertexIndex == 0u && IndirectOnly()))
                radiance += pathThroughput * SampleDirectLight(hit, surface, seed);

            if (vertexIndex >= (uint)numBounces)
                break;

            // Continuation sample: stochastic specular/diffuse selection,
            // pdf-cancelled throughput.
            float3 F = FresnelSchlick(surface.NdotV, surface.F0);
            float specularProb = SurfaceSpecularProb(surface);

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

            prevBouncePdf = PdfBsdfMixture(surface, specularProb, bounceDir);
            previousVertexPosition = hit.position;

#if RT_DEBUG_VIEWS
            // BounceHealth: classify a NaN bounce direction at this hit.
            if (debugMode == 2)
            {
                radiance += pathThroughput * BounceHealthColor(bounceDir, N, roughness);
                break;
            }
#endif

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