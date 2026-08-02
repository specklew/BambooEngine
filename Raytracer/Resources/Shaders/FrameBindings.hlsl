#ifndef FRAME_BINDINGS_HLSL
#define FRAME_BINDINGS_HLSL

// space0 is the frame layout: the resources every raytracing pass sees, at fixed
// registers, mirrored by FrameBindingLayout.h on the C++ side. Shared includes
// (RaytracingUtils, LightPool, BRDF) reference these by global register, which is
// why the layout is reserved and hand-curated rather than derived from usage.
//
// space1 is deliberately left empty: it is where per-pass bindings go.
//
// Everything declared here is bound for every pass whether it reads it or not, so
// keep the list short. Pass-scoped resources do not belong in this file.

#define MAX_TEXTURES 512

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
    // Object-to-world in DXR ObjectToWorld3x4() layout (transpose of the
    // row-vector world matrix). Lets raygen shaders reconstruct hits from the
    // VBuffer without hit-shader intrinsics.
    row_major float3x4 objectToWorld;
    float3 emissiveRadiance; // 0 = not emissive
    int    emissiveLightOffset; // -1 = not a light; else light-pool index of primitive 0
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

struct EmissiveTriangle
{
    float3 v0; float3 v1; float3 v2;
    float3 radiance;
    float  area;
    uint   instanceId;
};

struct LightPoolEntry
{
    uint  kind;      // 0 analytic, 1 emissive triangle
    uint  dataIndex;
    float power;
    float cdf;
};

RWTexture2D<float4> gOutput : register(u0);

RaytracingAccelerationStructure SceneBVH : register(t0);
ByteAddressBuffer g_vertices              : register(t1);
ByteAddressBuffer g_indices               : register(t2);

StructuredBuffer<GeometryInfo>     g_geometryInfo      : register(t3);
StructuredBuffer<InstanceInfo>     g_instanceInfo      : register(t4);
StructuredBuffer<EmissiveTriangle> g_emissiveTriangles : register(t5);
StructuredBuffer<LightData>        g_lightData         : register(t6);
StructuredBuffer<LightPoolEntry>   g_lightPool         : register(t7);

Texture2D g_skybox : register(t8);

// Last, so the bindless array can grow without moving anything above it.
Texture2D g_textures[MAX_TEXTURES] : register(t16);

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

#endif // FRAME_BINDINGS_HLSL
