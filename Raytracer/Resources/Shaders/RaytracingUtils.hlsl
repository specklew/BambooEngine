#ifndef RAYTRACING_UTILS_HLSL
#define RAYTRACING_UTILS_HLSL

#include "Random.hlsl"

#define MAX_TEXTURES 512
#define VERTEX_STRIDE 48 // float3 pos + float3 normal + float4 tangent + float2 uv

static const float MIN_ROUGHNESS = 0.04;
static const float RAY_TMIN = 0.01;
static const float RAY_TMAX = 1000.0;

// ---- Struct definitions ----

struct Payload
{
    float3 color;
    float3 throughput;
    uint bounceCount;
    uint seed;
};

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
    int normalTextureIndex;
    int roughnessTextureIndex;
    float metallicFactor;
    float roughnessFactor;
    float4 baseColorFactor;
};

struct LightData
{
    uint type; // 0 = directional, 1 = point, 2 = spot (not implemented)
    float3 position;
    float3 direction;
    float3 color;
    float intensity;
    float range;
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
    float4 tri_t0; // xyz = tangent, w = bitangent sign
    float4 tri_t1;
    float4 tri_t2;
    float3 tri_v0;
    float3 tri_v1;
    float3 tri_v2;
    float2 tri_uv0;
    float2 tri_uv1;
    float2 tri_uv2;

    // Hit point data
    float3 normal;
    float4 tangent; // xyz = tangent, w = bitangent sign
    float3 position;
    float2 uv;
};

// ---- Resource declarations ----

RWTexture2D<float4> gOutput : register(u0);
RaytracingAccelerationStructure SceneBVH : register(t0);

ByteAddressBuffer g_vertices : register(t1);
ByteAddressBuffer g_indices : register(t2);

StructuredBuffer<GeometryInfo> g_geometryInfo : register(t3);
StructuredBuffer<InstanceInfo> g_instanceInfo : register(t4);
StructuredBuffer<LightData> g_lightData : register(t6);

Texture2D g_textures[MAX_TEXTURES] : register(t7);

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

// ---- Vertex / index loading ----

uint3 Load3x32BitIndices(uint triangleIndex, uint indexOffset = 0)
{
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

float4 GetVertexFloat4Attribute(uint byteOffset)
{
    return asfloat(g_vertices.Load4(byteOffset));
}

HitData GetHitData(uint triangleIndex, uint vertexOffset, uint indexOffset, float2 barycentrics)
{
    HitData hit_data;

    hit_data.tri_indices = Load3x32BitIndices(triangleIndex, indexOffset) + vertexOffset;

    // Vertex stride: 48 bytes (float3 pos + float3 normal + float4 tangent + float2 uv)
    hit_data.tri_v0 = GetVertexFloat3Attribute(hit_data.tri_indices.x * VERTEX_STRIDE + 0);
    hit_data.tri_v1 = GetVertexFloat3Attribute(hit_data.tri_indices.y * VERTEX_STRIDE + 0);
    hit_data.tri_v2 = GetVertexFloat3Attribute(hit_data.tri_indices.z * VERTEX_STRIDE + 0);

    hit_data.tri_n0 = GetVertexFloat3Attribute(hit_data.tri_indices.x * VERTEX_STRIDE + 12);
    hit_data.tri_n1 = GetVertexFloat3Attribute(hit_data.tri_indices.y * VERTEX_STRIDE + 12);
    hit_data.tri_n2 = GetVertexFloat3Attribute(hit_data.tri_indices.z * VERTEX_STRIDE + 12);

    hit_data.tri_t0 = GetVertexFloat4Attribute(hit_data.tri_indices.x * VERTEX_STRIDE + 24);
    hit_data.tri_t1 = GetVertexFloat4Attribute(hit_data.tri_indices.y * VERTEX_STRIDE + 24);
    hit_data.tri_t2 = GetVertexFloat4Attribute(hit_data.tri_indices.z * VERTEX_STRIDE + 24);

    hit_data.tri_uv0 = GetVertexFloat2Attribute(hit_data.tri_indices.x * VERTEX_STRIDE + 40);
    hit_data.tri_uv1 = GetVertexFloat2Attribute(hit_data.tri_indices.y * VERTEX_STRIDE + 40);
    hit_data.tri_uv2 = GetVertexFloat2Attribute(hit_data.tri_indices.z * VERTEX_STRIDE + 40);

    hit_data.tri_normal = normalize(cross(hit_data.tri_v2 - hit_data.tri_v0, hit_data.tri_v1 - hit_data.tri_v0));

    float3 bary = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    hit_data.uv     = bary.x * hit_data.tri_uv0 + bary.y * hit_data.tri_uv1 + bary.z * hit_data.tri_uv2;
    hit_data.normal  = normalize(mul((float3x3)ObjectToWorld3x4(), normalize(bary.x * hit_data.tri_n0 + bary.y * hit_data.tri_n1 + bary.z * hit_data.tri_n2)));

    float3 interpTangent = normalize(bary.x * hit_data.tri_t0.xyz + bary.y * hit_data.tri_t1.xyz + bary.z * hit_data.tri_t2.xyz);
    hit_data.tangent = float4(normalize(mul((float3x3)ObjectToWorld3x4(), interpTangent)), hit_data.tri_t0.w);

    hit_data.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    return hit_data;
}

// ---- Ray utilities ----

inline void GenerateCameraRay(uint2 index, uint sampleIndex, out float3 origin, out float3 direction)
{
    float2 dims = float2(DispatchRaysDimensions().xy);
    float2 rand_offset = Random2D(sampleIndex) - 0.5;
    float2 d = (((index + 0.5f + rand_offset) / dims) * 2.f - 1.f);

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

// ---- Texture sampling ----

float4 SampleTexture(int texIdx, float2 uv)
{
    if (texIdx == -1)
        return float4(0, 0, 0, 0);
    return g_textures[NonUniformResourceIndex(texIdx)].SampleLevel(gsamAnisotropicWrap, uv, 0);
}

float3 SampleWorldSpaceNormal(HitData data)
{
    int normalTexIdx = g_instanceInfo[NonUniformResourceIndex(InstanceID())].normalTextureIndex;

    float3 normalTS = SampleTexture(normalTexIdx, data.uv).xyz * 2.0 - 1.0;

    float3 N = data.normal;
    float3 T = normalize(data.tangent.xyz - dot(data.tangent.xyz, N) * N);
    float3 B = cross(N, T) * data.tangent.w;
    float3x3 TBN = float3x3(T, B, N);

    return normalize(mul(normalTS, TBN));
}

float3 SampleTextureColor(HitData data)
{
    int texIdx = g_instanceInfo[NonUniformResourceIndex(InstanceID())].textureIndex;
    float4 col = SampleTexture(texIdx, data.uv);
    return col.rgb;
}

float2 SampleRoughnessMetallic(HitData data, float roughnessFactor, float metallicFactor)
{
    int rmTexIdx = g_instanceInfo[NonUniformResourceIndex(InstanceID())].roughnessTextureIndex;
    float4 mr = SampleTexture(rmTexIdx, data.uv);
    return float2(roughnessFactor * mr.g, metallicFactor * mr.b);
}

#endif // RAYTRACING_UTILS_HLSL