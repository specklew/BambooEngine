#ifndef GUIDED_PATH_TRACING_HLSL
#define GUIDED_PATH_TRACING_HLSL

// VXPG guided path tracing (stage A): two-sample MIS at the first bounce
// between BSDF sampling and a guide distribution. Guide is currently a
// uniform sphere — image must match vanilla PT modulo noise, which proves
// the MIS plumbing is unbiased. Stage B swaps the guide for the voxel
// irradiance distribution.
//
// Reuses scene bindings, BRDF helpers and the vanilla payload from raytracing.hlsl.
#include "raytracing.hlsl"

// ---- BSDF pdf evaluation (mixture of GGX + cosine, matches sampling) ----

float PdfGGX(SurfaceData s, float3 dir)
{
    float3 H = normalize(s.V + dir);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);
    float D = DistributionGGX(NdotH, s.roughness);
    return D * NdotH / (4.0 * VdotH);
}

float PdfCosine(SurfaceData s, float3 dir)
{
    return max(dot(s.N, dir), 0.0) / PI;
}

float PdfBsdf(SurfaceData s, float specularProb, float3 dir)
{
    if (dot(dir, s.N) <= 0.0)
        return 0.0;
    return specularProb * PdfGGX(s, dir) + (1.0 - specularProb) * PdfCosine(s, dir);
}

// Full BRDF eval consistent with the vanilla bounce estimator: GGX specular
// with the path-tracing Smith G (k = a^2/2, NOT the direct-lighting remap)
// plus Lambertian diffuse weighted by kD at NdotV — matches vanilla Hit()
// so the MIS estimator converges to the same image.
float3 EvalBsdfBounce(SurfaceData s, float3 dir)
{
    float NdotL = dot(s.N, dir);
    if (NdotL <= 0.0)
        return float3(0, 0, 0);
    NdotL = max(NdotL, EPSILON);

    float3 H = normalize(s.V + dir);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);

    float  D = DistributionGGX(NdotH, s.roughness);
    float  G = SmithG_GGX(s.NdotV, NdotL, s.roughness);
    float3 F = FresnelSchlick(VdotH, s.F0);
    float3 specular = (D * G * F) / (4.0 * s.NdotV * NdotL + EPSILON);

    // kD from Fresnel at NdotV — same as vanilla's path selection weighting
    float3 Fn = FresnelSchlick(s.NdotV, s.F0);
    float3 kD = (1.0 - Fn) * (1.0 - s.metallic);
    float3 diffuse = kD * s.albedo / PI;

    return specular + diffuse;
}

float3 SampleBsdfDir(SurfaceData s, float specularProb, float2 xi, out float pdf)
{
    float selector = frac(xi.x * 7.13 + xi.y * 3.97);
    float3 dir;
    if (selector < specularProb)
    {
        float3 H = ImportanceSampleGGX(xi, s.N, s.roughness);
        dir = reflect(-s.V, H);
    }
    else
    {
        dir = CosineSampleHemisphere(xi, s.N);
    }
    pdf = PdfBsdf(s, specularProb, dir);
    return dir;
}

// ---- Guide distribution (stage A: uniform sphere; stage B: voxel CDF) ----

float3 SampleGuideDir(float3 shadingPos, float2 xi, out float pdf)
{
    float z = 1.0 - 2.0 * xi.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * xi.y;
    pdf = 1.0 / (4.0 * PI);
    return float3(r * cos(phi), r * sin(phi), z);
}

float EvalGuidePdf(float3 shadingPos, float3 dir)
{
    return 1.0 / (4.0 * PI);
}

// ---- MIS weight (balance or power heuristic via guidingFlags bit 0) ----

float MisWeight(float wSelf, float wOther)
{
    if ((guidingFlags & 1u) != 0u)
    {
        wSelf  *= wSelf;
        wOther *= wOther;
    }
    return wSelf / (wSelf + wOther + EPSILON);
}

// ---- Continuation trace (deeper bounces, vanilla logic) ----

float3 TraceIndirect(float3 origin, float3 dir, inout uint seed)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    Payload p;
    p.color = float3(0, 0, 0);
    p.throughput = float3(1, 1, 1);
    p.bounceCount = 1;
    p.seed = seed;
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, p);
    seed = p.seed;
    return p.color;
}

// ---- Closest hit ----

[shader("closesthit")]
void GuidedHit(inout Payload payload : SV_RayPayload, in Attributes attr)
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

    if (payload.bounceCount >= numBounces)
    {
        payload.color = CalculateDirectLightning(hit, surface);
        return;
    }

    float3 F = FresnelSchlick(NdotV, surface.F0);
    float specularProb = (F.r + F.g + F.b) / 3.0;

    if (payload.bounceCount == 0)
    {
        // ---- First bounce: two-sample MIS between BSDF and guide ----
        // guidingFlags bits 1-2: debug view (0 = off, 1 = BSDF strategy only,
        // 2 = guide strategy only, 3 = MIS weight false-color)
        uint debugView = (guidingFlags >> 1) & 3u;

        float3 radiance = (debugView == 3u) ? float3(0, 0, 0)
                                            : CalculateDirectLightning(hit, surface);

        float misWeightB = 0.0;
        float misWeightG = 0.0;

        // BSDF strategy
        if (debugView != 2u)
        {
            float2 xi = Random2D(payload.seed);
            payload.seed = pcg_hash(payload.seed);

            float pdfB;
            float3 dir = SampleBsdfDir(surface, specularProb, xi, pdfB);
            if (pdfB > EPSILON && dot(dir, surface.N) > 0.0)
            {
                float3 f = EvalBsdfBounce(surface, dir);
                if (any(f > 0))
                {
                    float weight = MisWeight(pdfB, EvalGuidePdf(hit.position, dir));
                    misWeightB = weight;
                    if (debugView != 3u)
                    {
                        float NdotL = dot(surface.N, dir);
                        float3 incoming = TraceIndirect(hit.position, dir, payload.seed);
                        radiance += f * NdotL * incoming * weight / pdfB;
                    }
                }
            }
        }

        // Guide strategy
        if (debugView != 1u)
        {
            float2 xi = Random2D(payload.seed);
            payload.seed = pcg_hash(payload.seed);

            float pdfG;
            float3 dir = SampleGuideDir(hit.position, xi, pdfG);
            if (pdfG > EPSILON && dot(dir, surface.N) > 0.0)
            {
                float3 f = EvalBsdfBounce(surface, dir);
                if (any(f > 0))
                {
                    float weight = MisWeight(pdfG, PdfBsdf(surface, specularProb, dir));
                    misWeightG = weight;
                    if (debugView != 3u)
                    {
                        float NdotL = dot(surface.N, dir);
                        float3 incoming = TraceIndirect(hit.position, dir, payload.seed);
                        radiance += f * NdotL * incoming * weight / pdfG;
                    }
                }
            }
        }

        // MIS weight false-color: R = BSDF strategy weight, G = guide strategy
        // weight (at their respective sampled directions). Accumulates to the
        // mean weights — red where BSDF dominates, green where guide dominates.
        if (debugView == 3u)
            radiance = float3(misWeightB, misWeightG, 0);

        payload.color = radiance;
        return;
    }

    // ---- Deeper bounces: vanilla pdf-cancelled path tracing ----

    float2 xi = Random2D(payload.seed);
    payload.seed = pcg_hash(payload.seed);

    float pathSelector = frac(xi.x * 7.13 + xi.y * 3.97);

    float3 bounceDir;
    float3 throughput;

    if (pathSelector < specularProb)
    {
        float3 H = ImportanceSampleGGX(xi, N, roughness);
        bounceDir = reflect(-V, H);
        throughput = EvalSpecularBounce(surface, H, bounceDir);
        if (all(throughput == 0))
        {
            // Invalid bounce sample — keep the direct light at this vertex
            payload.color = CalculateDirectLightning(hit, surface);
            payload.bounceCount++;
            return;
        }
        throughput /= specularProb;
    }
    else
    {
        float3 kD = (1.0 - F) * (1.0 - metallic);
        bounceDir = CosineSampleHemisphere(xi, N);
        throughput = EvalDiffuseBounce(surface, kD, bounceDir);
        if (all(throughput == 0))
        {
            // Invalid bounce sample — keep the direct light at this vertex
            payload.color = CalculateDirectLightning(hit, surface);
            payload.bounceCount++;
            return;
        }
        throughput /= (1.0 - specularProb + EPSILON);
    }

    RayDesc ray;
    ray.Origin = hit.position;
    ray.Direction = bounceDir;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    payload.bounceCount++;
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

    float3 directHere = CalculateDirectLightning(hit, surface);
    float3 incoming = payload.color;
    payload.color = directHere + throughput * incoming;
    payload.throughput = throughput * payload.throughput;
}

#endif // GUIDED_PATH_TRACING_HLSL
