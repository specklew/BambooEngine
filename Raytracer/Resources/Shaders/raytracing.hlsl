#define MAX_TEXTURES 256

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

struct GeometryInfo
{
    uint vertexOffset;
    uint indexOffset;
};

struct InstanceInfo
{
    uint geometryIndex;
    int textureIndex;
};

struct TriangleData
{
    uint3 indices;
    float3 normal;
    float3 n0;
    float3 n1;
    float3 n2;
    float3 v0;
    float3 v1;
    float3 v2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
};

struct HitData
{
    // Triangle data (flattened to avoid nested struct issues)
    uint3 tri_indices;
    float3 tri_normal;
    float3 tri_n0;
    float3 tri_n1;
    float3 tri_n2;
    float3 tri_v0;
    float3 tri_v1;
    float3 tri_v2;
    float2 tri_uv0;
    float2 tri_uv1;
    float2 tri_uv2;
    
    // Hit point data
    float3 normal;
    float3 position;
    float2 uv;
};

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

ByteAddressBuffer g_vertices : register(t1);
ByteAddressBuffer g_indices : register(t2);

StructuredBuffer<GeometryInfo> g_geometryInfo : register(t3);
StructuredBuffer<InstanceInfo> g_instanceInfo : register(t4);
StructuredBuffer<float> g_random : register(t5);

Texture2D g_textures[MAX_TEXTURES] : register(t6);

SamplerState gsamPointWrap : register(s0);
SamplerState gsamPointClamp : register(s1);
SamplerState gsamLinearWrap : register(s2);
SamplerState gsamLinearClamp : register(s3);
SamplerState gsamAnisotropicWrap : register(s4);
SamplerState gsamAnisotropicClamp : register(s5);

cbuffer CameraParams : register(b0)
{
    float4x4 worldViewProj;
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
}

cbuffer Constants : register(b1)
{
    float time;
}

float rand_1_05(in float2 uv)
{
    float2 noise = (frac(sin(dot(uv ,float2(12.9898,78.233)*2.0)) * 43758.5453));
    return abs(noise.x + noise.y) * 0.5;
}

float3 RandomDirectionInHemisphere(float3 normal, float seed = 1.0f)
{
    float uv_1 = sin(g_random.Load(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().y) + seed + time);
    float uv_2 = cos(g_random.Load(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().y) + seed + time);
    float2 s1 = float2(uv_1, uv_2);
    float2 s2 = float2(cos(uv_1 + 1), sin(uv_2 + 2));
    float2 s3 = float2(sin(uv_2 + 1), cos(uv_1 + 2));
    float3 randomDirection = float3(rand_1_05(s1), rand_1_05(s2), rand_1_05(s3)) * 2.0f - 1.0f;
    if (dot(randomDirection, normal) < 0.0f)
        randomDirection = -randomDirection;
    return normalize(randomDirection);
}

uint3 Load3x32BitIndices(uint triangleIndex, uint indexOffset = 0)
{
    // byte to load = ((instance triangle) * 3 + (vert in triangle) + offset) * (bytes per index)
    uint3 indices;
    indices.x = g_indices.Load((triangleIndex * 3 + 0 + indexOffset) * 4);
    indices.y = g_indices.Load((triangleIndex * 3 + 1 + indexOffset) * 4);
    indices.z = g_indices.Load((triangleIndex * 3 + 2 + indexOffset) * 4);
    
    return indices;
}

float2 GetVertexFloat2Attribute(uint byteOffset)
{
    return asfloat(g_vertices.Load2(byteOffset));
}

float3 GetVertexFloat3Attribute(uint byteOffset)
{
    return asfloat(g_vertices.Load3(byteOffset));
}

HitData GetHitData(uint triangleIndex, uint vertexOffset, uint indexOffset, float2 barycentrics)
{
    HitData hit_data;
    
    hit_data.tri_indices = Load3x32BitIndices(triangleIndex, indexOffset) + vertexOffset;

    hit_data.tri_uv0 = GetVertexFloat2Attribute(hit_data.tri_indices.x * 32 + 24);
    hit_data.tri_uv1 = GetVertexFloat2Attribute(hit_data.tri_indices.y * 32 + 24);
    hit_data.tri_uv2 = GetVertexFloat2Attribute(hit_data.tri_indices.z * 32 + 24);

    hit_data.tri_v0 = GetVertexFloat3Attribute(hit_data.tri_indices.x * 32 + 0);
    hit_data.tri_v1 = GetVertexFloat3Attribute(hit_data.tri_indices.y * 32 + 0);
    hit_data.tri_v2 = GetVertexFloat3Attribute(hit_data.tri_indices.z * 32 + 0);

    hit_data.tri_n0 = GetVertexFloat3Attribute(hit_data.tri_indices.x * 32 + 12);
    hit_data.tri_n1 = GetVertexFloat3Attribute(hit_data.tri_indices.y * 32 + 12);
    hit_data.tri_n2 = GetVertexFloat3Attribute(hit_data.tri_indices.z * 32 + 12);

    hit_data.tri_normal = normalize(cross(hit_data.tri_v2 - hit_data.tri_v0, hit_data.tri_v1 - hit_data.tri_v0));

    float3 bary = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    
    hit_data.uv = bary.x * hit_data.tri_uv0 + bary.y * hit_data.tri_uv1 + bary.z * hit_data.tri_uv2;
    hit_data.normal = normalize(mul((float3x3)ObjectToWorld3x4(), normalize(bary.x * hit_data.tri_n0 + bary.y * hit_data.tri_n1 + bary.z * hit_data.tri_n2)));
    
    // Compute world-space hit position from the ray (vertex positions are in object space)
    hit_data.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    
    return hit_data;
}

inline void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    float2 dims = float2(DispatchRaysDimensions().xy);
    
    float2 d = (((index + 0.5f) / dims) * 2.f - 1.f);

    // Match the RayGen logic exactly
    origin = mul(viewI, float4(0, 0, 0, 1)).xyz;
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    direction = normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}


float3 RayPlaneIntersection(float3 planeOrigin, float3 planeNormal, float3 rayOrigin, float3 rayDirection)
{
    float t = dot(-planeNormal, rayOrigin - planeOrigin) / dot(planeNormal, rayDirection);
    return rayOrigin + rayDirection * t;
}

/*
    REF: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    From "Real-Time Collision Detection" by Christer Ericson
*/
float3 BarycentricCoordinates(float3 pt, float3 v0, float3 v1, float3 v2)
{
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 e2 = pt - v0;
    float d00 = dot(e0, e0);
    float d01 = dot(e0, e1);
    float d11 = dot(e1, e1);
    float d20 = dot(e2, e0);
    float d21 = dot(e2, e1);
    float denom = 1.0 / (d00 * d11 - d01 * d01);
    float v = (d11 * d20 - d01 * d21) * denom;
    float w = (d00 * d21 - d01 * d20) * denom;
    float u = 1.0 - v - w;
    return float3(u, v, w);
}

float3 SampleTextureColor(HitData data)
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    float3 ddxOrigin, ddxDirection, ddyOrigin, ddyDirection;
    GenerateCameraRay(launchIndex + uint2(1, 0), ddxOrigin, ddxDirection);
    GenerateCameraRay(launchIndex + uint2(0, 1), ddyOrigin, ddyDirection);

    float3 xOffsetPoint = RayPlaneIntersection(data.tri_v0, data.tri_normal, ddxOrigin, ddxDirection);
    float3 yOffsetPoint = RayPlaneIntersection(data.tri_v0, data.tri_normal, ddyOrigin, ddyDirection);

    float3 baryX = BarycentricCoordinates(xOffsetPoint, data.tri_v0, data.tri_v1, data.tri_v2);
    float3 baryY = BarycentricCoordinates(yOffsetPoint, data.tri_v0, data.tri_v1, data.tri_v2);

    float3x2 uvMat = float3x2(data.tri_uv0, data.tri_uv1, data.tri_uv2);
    float2 ddxUV = mul(baryX, uvMat) - data.uv;
    float2 ddyUV = mul(baryY, uvMat) - data.uv;
    
    float4 col = float4(0, 0, 0, 1);
    
    if (g_instanceInfo[InstanceID()].textureIndex != -1)
    {
        col = g_textures[g_instanceInfo[InstanceID()].textureIndex].SampleGrad(gsamAnisotropicWrap, data.uv, ddxUV, ddyUV);
    }

    return col;
}

[shader("raygeneration")] 
void RayGen() {
    // Initialize the ray payload
    HitInfo payload = {float4(0, 0, 0, 0)};

    // Get the location within the dispatched 2D grid of work items
    // (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;

    float3 origin, direction;
    GenerateCameraRay(launchIndex, origin, direction);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.01;
    ray.TMax = 1000;

    float4 col = float4(0, 0, 0, 0);
    for (int i = 0; i < 4; i++)
    {
        TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
        col += float4(payload.colorAndDistance.xyz, 1);
    }

    float3 color = col / 4.0f;
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
    //float4 color;
};

[shader("closesthit")] 
void Hit(inout HitInfo payload : SV_RayPayload, Attributes attr) 
{
    uint geometryIndex = g_instanceInfo[InstanceID()].geometryIndex;
    uint vertexOffset = g_geometryInfo[geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[geometryIndex].indexOffset;
    
    HitData hit_data = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);
    
    if (payload.colorAndDistance.z >= 3.0f) // already hit something closer, so ignore this hit
    {
        payload.colorAndDistance = float4(0, 0, 0, 3.0f);
        return;
    }
    
    float4 color = float4(0, 0, 0, 0);

    RayDesc ray;
    ray.Origin = hit_data.position;
    ray.Direction = RandomDirectionInHemisphere(hit_data.normal, 0);
    ray.TMin = 0.01;
    ray.TMax = 1000;

    payload.colorAndDistance.z += 1.0f; // increment bounce count in payload to limit number of bounces
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
    color += float4(payload.colorAndDistance.xyz, 0.0f);
    
    color.xyz *= SampleTextureColor(hit_data);
    
    payload.colorAndDistance = color;
}
