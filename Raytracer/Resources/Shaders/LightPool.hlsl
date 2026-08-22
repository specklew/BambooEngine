#ifndef LIGHT_POOL_HLSL
#define LIGHT_POOL_HLSL

// Unified NEE over the analytic + emissive-triangle pool. Selection pdf is the
// actual CDF sampled, so any positive power weighting stays unbiased.

float BalanceWeight(float pdfSelf, float pdfOtherA, float pdfOtherB)
{
    float denom = pdfSelf + pdfOtherA + pdfOtherB;
    return (pdfSelf > 0.0 && denom > 0.0) ? pdfSelf / denom : 0.0;
}

int SelectLightFromPool(float xi)
{
    int lo = 0, hi = int(lightPoolCount) - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (g_lightPool[mid].cdf < xi) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// NEE pdf (select x solid-angle) of generating the direction from shadingPos
// that hit emissive geometry at hitPos. 0 for non-emissive/back-face hits.
float PdfNeeTowardHit(float3 shadingPos, InstanceInfo hitInstance, uint hitPrimitiveId, float3 hitPos)
{
    if (hitInstance.emissiveLightOffset < 0 || lightPoolTotalPower <= 0.0)
        return 0.0;
    uint poolIndex = uint(hitInstance.emissiveLightOffset) + hitPrimitiveId;
    LightPoolEntry entry = g_lightPool[poolIndex];
    EmissiveTriangle tri = g_emissiveTriangles[entry.dataIndex];
    float3 toHit = hitPos - shadingPos;
    float distSq = dot(toHit, toHit);
    if (distSq <= 0.0 || tri.area <= 0.0)
        return 0.0;
    float3 lightNormal = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    float cosLight = dot(lightNormal, -normalize(toHit));
    if (cosLight <= 0.0)
        return 0.0;
    float pdfSelect = entry.power / lightPoolTotalPower;
    return pdfSelect * distSq / (cosLight * tri.area);
}

// One NEE sample from the pool, split from the MIS weight so a caller with a
// THIRD sampler at this vertex (the VXPG voxel guide) can fold its reverse pdf
// toward the sampled emitter point into the balance denominator for an exact
// sum-to-1. Identical sampling/occlusion and identical fixed RNG consumption
// (1 + 2) on every path. Outputs:
//   sampledKind: 0 = nothing (early-out, contribution 0),
//                1 = analytic delta light (contribution already finished, weight
//                    1 — the caller applies no further weight),
//                2 = area emitter (contributionOverPdfUnweighted is the raw
//                    brdf*Le*NdotL/pdfNee; the caller/wrapper applies the weight)
//   lightPoint: the sampled emitter point (area only; for the guide reverse pdf)
//   pdfNee: solid-angle NEE pdf (select x geometry)
//   pdfBsdfTowardLight: BSDF-mixture pdf toward the sampled direction
//   irradianceOverPdf: the same sample WITHOUT the BRDF factor and WITHOUT any MIS
//     weight — a single-sample estimate of the cosine-weighted incident radiance, i.e.
//     E(x) in the VXPG paper's Eq. 5. The guide is fitted to this; the image is not.
void SampleDirectLightComponents(HitData hit, SurfaceData surface, inout uint seed,
                                 out uint sampledKind, out float3 lightPoint,
                                 out float pdfNee, out float pdfBsdfTowardLight,
                                 out float3 contributionOverPdfUnweighted,
                                 out float3 irradianceOverPdf)
{
    sampledKind = 0u;
    lightPoint = float3(0, 0, 0);
    pdfNee = 0.0;
    pdfBsdfTowardLight = 0.0;
    contributionOverPdfUnweighted = float3(0, 0, 0);
    irradianceOverPdf = float3(0, 0, 0);

    float selectXi = Random1D(seed);
    seed = pcg_hash(seed);
    float2 areaXi = Random2D(seed);
    seed = pcg_hash(seed);

    if (lightPoolTotalPower <= 0.0 || lightPoolCount == 0)
        return;

    int poolIndex = SelectLightFromPool(selectXi);
    LightPoolEntry entry = g_lightPool[poolIndex];
    float pdfSelect = entry.power / lightPoolTotalPower;
    if (pdfSelect <= 0.0)
        return;

    if (entry.kind == 0) // analytic delta light: existing path, weight 1
    {
        LightData light = g_lightData[entry.dataIndex];
        float3 L = GetShadowRayDirection(hit.position, light);
        float visibility = TraceShadow(hit.position + surface.N * EPSILON, light);
        if (visibility <= 0.0)
            return;
        float atten = GetLightAttenuation(hit.position, light);
        float3 brdf = EvalDirectBRDF(surface, L);
        const float3 incidentOverPdf = light.color * light.intensity * atten * visibility
             * max(dot(surface.N, L), 0.0) / pdfSelect;
        contributionOverPdfUnweighted = brdf * incidentOverPdf;
        irradianceOverPdf = incidentOverPdf;
        sampledKind = 1u; // delta: finished, weight 1
        return;
    }

    // Emissive triangle: uniform area sample, solid-angle pdf, MIS vs BSDF.
    EmissiveTriangle tri = g_emissiveTriangles[entry.dataIndex];
    if (tri.area <= 0.0)
        return;
    float sqrtU = sqrt(max(areaXi.x, 0.0));
    float b0 = 1.0 - sqrtU;
    float b1 = areaXi.y * sqrtU;
    float3 lp = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * (1.0 - b0 - b1);

    float3 toLight = lp - hit.position;
    float distSq = dot(toLight, toLight);
    float dist = sqrt(distSq);
    if (dist <= EPSILON)
        return;
    float3 L = toLight / dist;

    float NdotL = dot(surface.N, L);
    float3 lightNormal = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    float cosLight = dot(lightNormal, -L);
    if (NdotL <= 0.0 || cosLight <= 0.0) // below horizon or emitter back face
        return;

    if (!TraceShadowSegment(hit.position + surface.N * EPSILON, L, dist))
        return;

    float pdfNeeLocal = pdfSelect * distSq / (cosLight * tri.area);
    float3 brdf = EvalPathBRDF(surface, L); // same BRDF the BSDF/guide strategies see (ADR 0016 M4)

    lightPoint = lp;
    pdfNee = pdfNeeLocal;
    pdfBsdfTowardLight = PdfBsdfMixture(surface, SurfaceSpecularProb(surface), L);
    contributionOverPdfUnweighted = brdf * tri.radiance * NdotL / pdfNeeLocal;
    // No MIS weight here: E is the whole direct irradiance, not NEE's share of it.
    irradianceOverPdf = tri.radiance * NdotL / pdfNeeLocal;
    sampledKind = 2u; // area: caller applies the MIS weight
}

// Thin two-way wrapper: bit-identical to the pre-split SampleDirectLight for
// every caller without a third sampler (PT, injection, deep/tail vertices). The
// area branch reproduces brdf*Le*NdotL*BalanceWeight(pdfNee,pdfBsdf,0)/pdfNee.
float3 SampleDirectLight(HitData hit, SurfaceData surface, inout uint seed,
                        out float3 irradianceOverPdf)
{
    uint sampledKind;
    float3 lightPoint;
    float pdfNee, pdfBsdfTowardLight;
    float3 contributionOverPdfUnweighted;
    SampleDirectLightComponents(hit, surface, seed, sampledKind, lightPoint,
                                pdfNee, pdfBsdfTowardLight, contributionOverPdfUnweighted,
                                irradianceOverPdf);
    if (sampledKind == 2u)
        return contributionOverPdfUnweighted * BalanceWeight(pdfNee, pdfBsdfTowardLight, 0.0);
    return contributionOverPdfUnweighted; // delta (weight 1) or zero early-out
}

float3 SampleDirectLight(HitData hit, SurfaceData surface, inout uint seed)
{
    float3 unusedIrradiance;
    return SampleDirectLight(hit, surface, seed, unusedIrradiance);
}

#endif
