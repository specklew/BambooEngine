#ifndef GUIDED_PATH_TRACING_HLSL
#define GUIDED_PATH_TRACING_HLSL

// VXPG guided path tracing (stage B): two-sample MIS at the first bounce
// between BSDF sampling and the voxel irradiance distribution. A voxel is
// picked by binary-searching a CDF over compacted nonzero-irradiance voxels
// (built by VoxelGuidingBuildPass), then a direction is cone-sampled toward
// that voxel's bounding sphere ("voxel2sphere"). The guided contribution is
// gated to hits inside the chosen voxel's AABB, mirroring the reference
// integrator (vxguiding-gi.slang, strategy 4).
//
// Falls back to a uniform-sphere guide when no guiding data exists yet
// (e.g. injection disabled), which is exactly the verified stage-A path.
//
// Reuses scene bindings and BRDF helpers from raytracing.hlsl.
#include "raytracing.hlsl"

#define VOXEL_GUIDING_CAPACITY 131072

// ---- VXPG guiding resources ----

RWTexture3D<uint> gVoxIrradiance : register(u1);
RWTexture3D<uint> gVoxVplCount   : register(u2);

// [0] = compacted voxel count, [1] = asuint(total weight)
RWStructuredBuffer<uint>  gVoxCounters     : register(u3);
RWStructuredBuffer<uint>  gVoxCompactIds   : register(u4);
RWStructuredBuffer<float> gVoxCdf          : register(u5);
// voxelID (flat) -> compactID, sentinel -1 (built by VoxelGuidingBuildPass).
RWStructuredBuffer<int>   gVoxInverseIndex : register(u6);

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

float UnpackIrradiance(uint packed)
{
    return float(packed) / 100.0f;
}

// ---- Payload (carries first-hit position for guide pdf evaluation) ----

struct GuidedPayload
{
    float3 color;
    float3 hitPos;   // position of the FIRST hit of the traced ray
    uint   hitFlag;  // 1 = ray hit geometry
    uint   bounceCount;
    uint   seed;
};

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

// ---- Voxel guide distribution ----

bool GuidingDataValid(out uint n, out float total)
{
    n = min(gVoxCounters[0], VOXEL_GUIDING_CAPACITY);
    total = asfloat(gVoxCounters[1]);
    return n > 0u && total > 0.0f;
}

// Cone pdf toward the bounding sphere of voxel v as seen from shadingPos.
// Returns the solid-angle pdf; handles the "inside the sphere" case as a
// full-sphere distribution.
float ConePdfTowardVoxel(float3 shadingPos, int3 v, out float3 center, out float cosThetaMax)
{
    center = voxGridMin + (float3(v) + 0.5f) * voxVoxelSize;
    const float radius = voxVoxelSize * 0.8660254f; // half voxel diagonal
    const float dist = length(center - shadingPos);

    if (dist <= radius)
    {
        cosThetaMax = -1.0f;
        return 1.0f / (4.0f * PI);
    }

    cosThetaMax = sqrt(max(0.0f, 1.0f - (radius * radius) / (dist * dist)));
    return 1.0f / (2.0f * PI * (1.0f - cosThetaMax) + EPSILON);
}

// Probability of the guide picking the voxel containing worldPos.
float VoxelSelectionProb(float3 worldPos, float total, out int3 v)
{
    v = int3(floor((worldPos - voxGridMin) / voxVoxelSize));
    if (any(v < 0) || any(v >= int(voxGridDim)))
        return 0.0f;

    uint count = gVoxVplCount[v];
    if (count == 0u)
        return 0.0f;

    float w = UnpackIrradiance(gVoxIrradiance[v]) / float(count);
    if (w <= 0.0f)
        return 0.0f;

    return w / total;
}

// Guide pdf of a BSDF-sampled ray, evaluated at its hit position: probability
// of selecting the hit voxel times the cone pdf toward it (reference:
// PdfVoxelGuiding with flat/no-tree voxel selection).
float EvalGuidePdf(float3 shadingPos, float3 hitPos, float total)
{
    int3 v;
    float pVoxel = VoxelSelectionProb(hitPos, total, v);
    if (pVoxel <= 0.0f)
        return 0.0f;

    float3 center;
    float cosThetaMax;
    float pdfDir = ConePdfTowardVoxel(shadingPos, v, center, cosThetaMax);
    return pVoxel * pdfDir;
}

float3 SampleCone(float3 axis, float cosThetaMax, float2 xi)
{
    float cosTheta = 1.0f - xi.x * (1.0f - cosThetaMax);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * PI * xi.y;

    float3 local = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    float3 T, B;
    BuildONB(axis, T, B);
    return TangentToWorld(local, axis, T, B);
}

// Samples the voxel guide: pick a voxel by CDF binary search, cone-sample a
// direction toward its bounding sphere. Outputs the chosen voxel's AABB for
// the semi-NEE gate.
float3 SampleVoxelGuideDir(
    float3 shadingPos, uint n, float total, float selectXi, float2 coneXi,
    out float pdf, out float3 aabbMin, out float3 aabbMax)
{
    // Binary search: smallest k with cdf[k] >= target
    const float target = selectXi * total;
    uint lo = 0u, hi = n - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2u;
        if (gVoxCdf[mid] < target) lo = mid + 1u;
        else                       hi = mid;
    }
    const uint k = lo;

    const float cdfLo = (k > 0u) ? gVoxCdf[k - 1u] : 0.0f;
    const float pVoxel = max(gVoxCdf[k] - cdfLo, 0.0f) / total;

    const uint flatId = gVoxCompactIds[k];
    int3 v;
    v.x = int(flatId % voxGridDim);
    v.y = int((flatId / voxGridDim) % voxGridDim);
    v.z = int(flatId / (voxGridDim * voxGridDim));

    aabbMin = voxGridMin + float3(v) * voxVoxelSize;
    aabbMax = aabbMin + voxVoxelSize;

    float3 center;
    float cosThetaMax;
    const float pdfDir = ConePdfTowardVoxel(shadingPos, v, center, cosThetaMax);

    float3 dir;
    if (cosThetaMax < -0.5f) // shading point inside voxel bounding sphere
    {
        float z = 1.0f - 2.0f * coneXi.x;
        float r = sqrt(max(0.0f, 1.0f - z * z));
        float phi = 2.0f * PI * coneXi.y;
        dir = float3(r * cos(phi), r * sin(phi), z);
    }
    else
    {
        dir = SampleCone(normalize(center - shadingPos), cosThetaMax, coneXi);
    }

    pdf = pVoxel * pdfDir;
    return dir;
}

// Stage-A fallback: uniform sphere
float3 SampleUniformSphereDir(float2 xi, out float pdf)
{
    float z = 1.0f - 2.0f * xi.x;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI * xi.y;
    pdf = 1.0f / (4.0f * PI);
    return float3(r * cos(phi), r * sin(phi), z);
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

float3 TraceIndirect(float3 origin, float3 dir, inout uint seed, out float3 hitPos, out bool didHit)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    GuidedPayload p;
    p.color = float3(0, 0, 0);
    p.hitPos = float3(0, 0, 0);
    p.hitFlag = 0;
    p.bounceCount = 1;
    p.seed = seed;
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, p);
    seed = p.seed;
    hitPos = p.hitPos;
    didHit = (p.hitFlag != 0u);
    return p.color;
}

// ---- Ray generation ----

[shader("raygeneration")]
void GuidedRayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    uint pixelId = launchIndex.x + launchIndex.y * dims.x;

    float3 accumulated = float3(0, 0, 0);
    for (uint i = 0; i < (uint)samplesPerPixel; i++)
    {
        uint seed = pcg_hash(pixelId ^ (i * 2654435761u) ^ (frameIndex * 805459861u));

        float3 origin, direction;
        GenerateCameraRay(launchIndex, seed, origin, direction);
        seed = pcg_hash(seed);

        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = RAY_TMIN;
        ray.TMax = RAY_TMAX;

        GuidedPayload payload;
        payload.color = float3(0, 0, 0);
        payload.hitPos = float3(0, 0, 0);
        payload.hitFlag = 0;
        payload.bounceCount = 0;
        payload.seed = seed;
        TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, payload);
        accumulated += payload.color;
    }

    gOutput[launchIndex] = min(float4(accumulated / samplesPerPixel, 1.0), 100.0f);
}

// ---- Miss ----

[shader("miss")]
void GuidedMiss(inout GuidedPayload payload : SV_RayPayload)
{
    float3 dir = normalize(WorldRayDirection());
    float u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = -asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    payload.color = g_skybox.SampleLevel(gsamLinearWrap, float2(u, v), 0).rgb;
    payload.hitFlag = 0;
}

// ---- Any hit (alpha cutout) ----

[shader("anyhit")]
void GuidedAnyHit(inout GuidedPayload payload : SV_RayPayload, in Attributes attr)
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
void GuidedHit(inout GuidedPayload payload : SV_RayPayload, in Attributes attr)
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

    float3 geometricN = normalize(mul((float3x3)ObjectToWorld3x4(), hit.tri_normal));
    if (dot(geometricN, V) < 0.0)
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

    if (payload.bounceCount >= numBounces)
    {
        payload.color = CalculateDirectLightning(hit, surface);
        payload.hitPos = hit.position;
        payload.hitFlag = 1;
        return;
    }

    float3 F = FresnelSchlick(NdotV, surface.F0);
    float specularProb = (F.r + F.g + F.b) / 3.0;

    if (payload.bounceCount == 0)
    {
        // ---- First bounce: two-sample MIS between BSDF and voxel guide ----
        // guidingFlags bits 1-3: debug view (0 = off, 1 = BSDF strategy only,
        // 2 = guide strategy only, 3 = MIS weight false-color,
        // 4 = guided sample acceptance: green/red/blue)
        uint debugView = (guidingFlags >> 1) & 7u;

        // View 5: inverse-index round-trip. For the primary hit's voxel, look up
        // gInverseIndex -> compactID and confirm gCompactIds maps back to the same
        // voxel. green = consistent, red = mismatch (bug), black = inactive voxel.
        if (debugView == 5u)
        {
            int3 v = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
            float3 col = float3(0, 0, 0);
            if (all(v >= 0) && all(v < int(voxGridDim)))
            {
                uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
                int ci = gVoxInverseIndex[flatId];
                if (ci >= 0)
                    col = (gVoxCompactIds[ci] == flatId) ? float3(0, 1, 0) : float3(1, 0, 0);
            }
            payload.color = col;
            payload.hitPos = hit.position;
            payload.hitFlag = 1;
            return;
        }

        uint n;
        float total;
        const bool guidingActive = GuidingDataValid(n, total);

        float3 radiance = (debugView >= 3u) ? float3(0, 0, 0)
                                            : CalculateDirectLightning(hit, surface);

        float misWeightB = 0.0;
        float misWeightG = 0.0;
        // Acceptance classification (view 4): 0 = pre-trace reject (blue),
        // 1 = traced but gate-rejected (red), 2 = accepted (green)
        uint guideOutcome = 0u;

        // BSDF strategy
        if (debugView != 2u && debugView != 4u)
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
                    float3 hitPos;
                    bool didHit;
                    float3 incoming = TraceIndirect(hit.position, dir, payload.seed, hitPos, didHit);

                    // Guide pdf at the BSDF sample: evaluated at the ray's hit
                    // voxel (0 for misses — the guide only targets voxels)
                    float pdfGAtDir;
                    if (guidingActive)
                        pdfGAtDir = didHit ? EvalGuidePdf(hit.position, hitPos, total) : 0.0;
                    else
                        pdfGAtDir = 1.0 / (4.0 * PI); // stage-A uniform fallback

                    float weight = MisWeight(pdfB, pdfGAtDir);
                    misWeightB = weight;
                    if (debugView != 3u)
                    {
                        float NdotL = dot(surface.N, dir);
                        radiance += f * NdotL * incoming * weight / pdfB;
                    }
                }
            }
        }

        // Guide strategy
        if (debugView != 1u)
        {
            float2 selectXi = Random2D(payload.seed);
            payload.seed = pcg_hash(payload.seed);
            float2 coneXi = Random2D(payload.seed);
            payload.seed = pcg_hash(payload.seed);

            float pdfG;
            float3 dir;
            float3 aabbMin = float3(-1e30f, -1e30f, -1e30f);
            float3 aabbMax = float3( 1e30f,  1e30f,  1e30f);
            if (guidingActive)
                dir = SampleVoxelGuideDir(hit.position, n, total, selectXi.x, coneXi, pdfG, aabbMin, aabbMax);
            else
                dir = SampleUniformSphereDir(coneXi, pdfG);

            if (pdfG > EPSILON && dot(dir, surface.N) > 0.0)
            {
                float3 f = EvalBsdfBounce(surface, dir);
                if (any(f > 0))
                {
                    float3 hitPos;
                    bool didHit;
                    float3 incoming = TraceIndirect(hit.position, dir, payload.seed, hitPos, didHit);

                    // Semi-NEE gate: the claimed pdf belongs to the chosen
                    // voxel, so only count hits inside its AABB (reference
                    // does the same). Uniform fallback gates on nothing.
                    bool accepted = guidingActive
                        ? (didHit && all(hitPos >= aabbMin) && all(hitPos <= aabbMax))
                        : true;

                    guideOutcome = accepted ? 2u : 1u;

                    if (accepted)
                    {
                        float weight = MisWeight(pdfG, PdfBsdf(surface, specularProb, dir));
                        misWeightG = weight;
                        if (debugView < 3u)
                        {
                            float NdotL = dot(surface.N, dir);
                            radiance += f * NdotL * incoming * weight / pdfG;
                        }
                    }
                }
            }
        }

        // MIS weight false-color: R = BSDF strategy weight, G = guide strategy
        // weight (at their respective sampled directions).
        if (debugView == 3u)
            radiance = float3(misWeightB, misWeightG, 0);

        // Guided sample acceptance: green = accepted (hit inside chosen voxel),
        // red = traced but gate-rejected, blue = rejected before tracing
        // (below horizon / zero pdf / zero BRDF).
        if (debugView == 4u)
        {
            if (guideOutcome == 2u)      radiance = float3(0, 1, 0);
            else if (guideOutcome == 1u) radiance = float3(1, 0, 0);
            else                         radiance = float3(0, 0, 1);
        }

        payload.color = radiance;
        payload.hitPos = hit.position;
        payload.hitFlag = 1;
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
            payload.hitPos = hit.position;
            payload.hitFlag = 1;
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
            payload.hitPos = hit.position;
            payload.hitFlag = 1;
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

    // Restore this vertex as the ray's first hit (recursion overwrote it)
    payload.hitPos = hit.position;
    payload.hitFlag = 1;
}

#endif // GUIDED_PATH_TRACING_HLSL
