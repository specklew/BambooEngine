struct HitInfo
{
    float4 colorAndDistance;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes
{
    float2 barycentrics;
};

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

#include "BRDF.hlsl"
#include "Random.hlsl"
#include "RaytracingUtils.hlsl"

[shader("raygeneration")]
void RayGen() {
    HitInfo payload = {float4(0, 0, 0, 0)};

    uint2 launchIndex = DispatchRaysIndex().xy;

    float3 origin, direction;
    GenerateCameraRay(launchIndex, origin, direction);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.01;
    ray.TMax = 1000;

    float4 col = float4(0, 0, 0, 0);
    for (int i = 0; i < 16; i++)
    {
        TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
        col += float4(payload.colorAndDistance.xyz, 1);
    }

    float3 color = col / 16.0f;
    gOutput[launchIndex] = float4(color, 1.f);
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload)
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);

    float ramp = launchIndex.y / dims.y;
    payload.colorAndDistance = float4(0.7f, 0.8f, 1.0f - 0.1f*ramp, -1.0f);
}

struct STriVertex
{
    float3 vertex;
};

[shader("closesthit")]
void Hit(inout HitInfo payload : SV_RayPayload, Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;

    HitData hit_data = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    // Compute UV gradients once for all texture samples
    float2 ddxUV, ddyUV;
    ComputeUVGradients(hit_data, ddxUV, ddyUV);

    // Sample material properties
    float3 albedo = SampleTextureColor(hit_data, ddxUV, ddyUV) * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(hit_data, ddxUV, ddyUV, instance.roughnessFactor, instance.metallicFactor);
    float roughness = max(rm.x, 0.04); // clamp to avoid singularities
    float metallic = rm.y;

    if (payload.colorAndDistance.z >= 2.0f)
    {
        payload.colorAndDistance = float4(0.0f, 0.0f, 0.0f, 3.0f);
        return;
    }

    float3 N = SampleWorldSpaceNormal(hit_data, ddxUV, ddyUV);
    float3 V = -WorldRayDirection();
    float NdotV = max(dot(N, V), 0.001);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // Generate random numbers for this bounce
    float2 xi = Random2D(payload.colorAndDistance.z);

    // Fresnel at normal incidence to decide specular vs diffuse path
    float3 F = FresnelSchlick(NdotV, F0);
    float specularWeight = (F.r + F.g + F.b) / 3.0;

    float3 bounceDir;
    float3 throughput;

    // Stochastic path selection: specular or diffuse
    float pathSelector = frac(xi.x * 7.13 + xi.y * 3.97);
    if (pathSelector < specularWeight)
    {
        // Specular path: GGX importance sampling
        float3 H = ImportanceSampleGGX(xi, N, roughness);
        bounceDir = reflect(-V, H);

        if (dot(bounceDir, N) <= 0.0)
        {
            payload.colorAndDistance = float4(0, 0, 0, payload.colorAndDistance.w);
            return;
        }

        float3 Fh = FresnelSchlick(max(dot(V, H), 0.0), F0);
        // For specular: throughput = F * G_vis (simplified for importance-sampled GGX)
        // The GGX PDF cancels D*NdotH, leaving G * VdotH / (NdotV * NdotH)
        float NdotH = max(dot(N, H), 0.001);
        float NdotL = max(dot(N, bounceDir), 0.001);
        float VdotH = max(dot(V, H), 0.001);

        // Smith G term for GGX
        float a = roughness * roughness;
        float k = (a * a) / 2.0;
        float G1V = NdotV / (NdotV * (1.0 - k) + k);
        float G1L = NdotL / (NdotL * (1.0 - k) + k);
        float G = G1V * G1L;

        throughput = Fh * G * VdotH / (NdotV * NdotH + 0.0001);
        throughput /= specularWeight; // MIS: divide by selection probability
    }
    else
    {
        // Diffuse path: cosine-weighted hemisphere sampling
        bounceDir = CosineSampleHemisphere(xi, N);

        if (dot(bounceDir, N) <= 0.0)
        {
            payload.colorAndDistance = float4(0, 0, 0, payload.colorAndDistance.w);
            return;
        }

        // Diffuse BRDF: albedo/PI, cosine PDF: NdotL/PI -> they cancel to just albedo
        float3 kD = (1.0 - F) * (1.0 - metallic);
        throughput = kD * albedo;
        throughput /= (1.0 - specularWeight + 0.0001); // MIS: divide by selection probability
    }

    // Trace the bounce ray
    RayDesc ray;
    ray.Origin = hit_data.position;
    ray.Direction = bounceDir;
    ray.TMin = 0.01;
    ray.TMax = 1000;

    payload.colorAndDistance.z += 1.0f;
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

    float3 incoming = payload.colorAndDistance.xyz;
    payload.colorAndDistance = float4(throughput * incoming, payload.colorAndDistance.w);
}