#include "BRDF.hlsl"
#include "Random.hlsl"
#include "RaytracingUtils.hlsl"
#include "raytracing.shadow.hlsl"
#include "passConstants.hlsl"

// ---- Ray generation ----

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

        float3 origin, direction;
        GenerateCameraRay(launchIndex, seed, origin, direction);
        seed = pcg_hash(seed);  // advance: camera ray and bounce 0 now use different xi

        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = RAY_TMIN;
        ray.TMax = RAY_TMAX;

        Payload payload;
        payload.color = float3(0, 0, 0);
        payload.throughput = float3(1, 1, 1);
        payload.bounceCount = 0;
        payload.seed = seed;
        TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
        accumulated += payload.color;
    }

    gOutput[launchIndex] = float4(accumulated / samplesPerPixel, 1.0);
}

// ---- Miss ----

[shader("miss")]
void Miss(inout Payload payload : SV_RayPayload)
{
    float ramp = WorldRayDirection().y;
    float3 baseColor = float3(0.5, 0.7, 0.9);
    payload.color = baseColor + ramp * 0.2;
}

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
    float NdotL = max(dot(s.N, bounceDir), 0.1);
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
    float NdotL = max(dot(s.N, L), 0.1);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);

    float3 F = FresnelSchlick(VdotH, s.F0);
    float  G = GeometrySmith(s.NdotV, NdotL, s.roughness);
    float  D = DistributionGGX(NdotH, s.roughness);

    return min((F * G * D) / (4.0 * s.NdotV * NdotL + EPSILON), float3(1, 1, 1));
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

[shader("closesthit")]
void Hit(inout Payload payload : SV_RayPayload, Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    // Material
    float3 albedo = SampleTextureColor(hit) * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(hit, instance.roughnessFactor, instance.metallicFactor);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;

    // Shading vectors
    float3 N = SampleWorldSpaceNormal(hit);
    float3 V = -WorldRayDirection();
    float NdotV = max(dot(N, V), 0.1);

    // Build surface data
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

    // Stochastic path selection — use payload.seed directly, then advance it
    float2 xi = Random2D(payload.seed);
    payload.seed = pcg_hash(payload.seed);  // advance: next bounce gets a different xi

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
            payload.color = float3(0, 0, 0);
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
            payload.color = float3(0, 0, 0);
            payload.bounceCount++;
            return;
        }
        throughput /= (1.0 - specularProb + EPSILON);
    }

    // Clamp throughput to suppress firefly samples from high-variance paths
    throughput = min(throughput, float3(5.0, 5.0, 5.0));

    // Trace bounce ray
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

