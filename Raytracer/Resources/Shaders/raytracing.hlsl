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

    float3 origin, direction;
    GenerateCameraRay(launchIndex, origin, direction);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    float3 accumulated = float3(0, 0, 0);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++)
    {
        Payload payload = { float4(0, 0, 0, 0), 0 };
        TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
        accumulated += payload.color.xyz;
    }

    gOutput[launchIndex] = float4(accumulated / SAMPLES_PER_PIXEL, 1.0);
}

// ---- Miss ----

[shader("miss")]
void Miss(inout Payload payload : SV_RayPayload)
{
    float2 dims = float2(DispatchRaysDimensions().xy);
    float ramp = DispatchRaysIndex().y / dims.y;
    payload.color = float4(0.7, 0.8, 1.0 - 0.1 * ramp, -1.0);
}

// ---- Closest hit ----

float GetBounceDepth(Payload payload)   { return payload.color.z; }

// Importance-sampled GGX specular throughput.
// The GGX PDF (D * NdotH / (4 * VdotH)) cancels D from the microfacet BRDF,
// leaving: throughput = F * G * VdotH / (NdotV * NdotH)
float3 EvalSpecularBounce(float2 xi, float3 N, float3 V, float NdotV, float3 F0, float roughness, out float3 bounceDir)
{
    float3 H = ImportanceSampleGGX(xi, N, roughness);
    bounceDir = reflect(-V, H);

    if (dot(bounceDir, N) <= 0.0)
        return float3(0, 0, 0);

    float NdotH = max(dot(N, H), EPSILON);
    float NdotL = max(dot(N, bounceDir), EPSILON);
    float VdotH = max(dot(V, H), EPSILON);

    float3 F = FresnelSchlick(VdotH, F0);
    float G = SmithG_GGX(NdotV, NdotL, roughness);

    return F * G * VdotH / (NdotV * NdotH + EPSILON);
}

// Cosine-weighted diffuse throughput.
// Lambertian BRDF (albedo/PI) divided by cosine PDF (NdotL/PI) cancels to albedo.
float3 EvalDiffuseBounce(float2 xi, float3 N, float3 kD, float3 albedo, out float3 bounceDir)
{
    bounceDir = CosineSampleHemisphere(xi, N);

    if (dot(bounceDir, N) <= 0.0)
        return float3(0, 0, 0);

    return kD * albedo;
}

[shader("closesthit")]
void Hit(inout Payload payload : SV_RayPayload, Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);
    
    float max_visibility = 0.0f;
    for (int i = 0; i < numLights; i++)
    {
        LightData light;
        light = g_lightData[i];
        max_visibility += TraceShadow(hit.position, light);
    }

    float intensity = max_visibility / numLights;
    payload.color = float4(intensity,intensity,intensity,0);

    return;
    
    float bounceDepth = GetBounceDepth(payload);
    if (bounceDepth >= MAX_BOUNCES)
    {
        payload.color = float4(0, 0, 0, bounceDepth + 1.0);
        return;
    }

    // Fetch geometry
    //InstanceInfo instance = g_instanceInfo[InstanceID()];
    //uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    ///uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    //HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    // UV gradients for anisotropic texture filtering
    float2 ddxUV, ddyUV;
    ComputeUVGradients(hit, ddxUV, ddyUV);

    // Material
    float3 albedo = SampleTextureColor(hit, ddxUV, ddyUV) * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(hit, ddxUV, ddyUV, instance.roughnessFactor, instance.metallicFactor);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;

    // Shading vectors
    float3 N = SampleWorldSpaceNormal(hit, ddxUV, ddyUV);
    float3 V = -WorldRayDirection();
    float NdotV = max(dot(N, V), EPSILON);

    // Fresnel at view angle determines specular vs diffuse probability
    float3 F0 = lerp(DIELECTRIC_F0, albedo, metallic);
    float3 F = FresnelSchlick(NdotV, F0);
    float specularProb = (F.r + F.g + F.b) / 3.0;

    // Stochastic path selection
    float2 xi = Random2D(bounceDepth);
    float pathSelector = frac(xi.x * 7.13 + xi.y * 3.97);

    float3 bounceDir;
    float3 throughput;

    if (pathSelector < specularProb)
    {
        throughput = EvalSpecularBounce(xi, N, V, NdotV, F0, roughness, bounceDir);
        if (all(throughput == 0))
        {
            payload.color = float4(0, 0, 0, payload.color.w);
            return;
        }
        throughput /= specularProb;
    }
    else
    {
        float3 kD = (1.0 - F) * (1.0 - metallic);
        throughput = EvalDiffuseBounce(xi, N, kD, albedo, bounceDir);
        if (all(throughput == 0))
        {
            payload.color = float4(0, 0, 0, payload.color.w);
            return;
        }
        throughput /= (1.0 - specularProb + EPSILON);
    }

    // Trace bounce ray
    RayDesc ray;
    ray.Origin = hit.position;
    ray.Direction = bounceDir;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    payload.color.z += 1.0;
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

    float3 incoming = payload.color.xyz;
    payload.color = float4(throughput * incoming, payload.color.w);
}

