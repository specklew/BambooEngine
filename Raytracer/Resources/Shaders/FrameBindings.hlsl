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

#include "FrameBindingRegisters.h"

#define MAX_TEXTURES FRAME_MAX_TEXTURES

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

RWTexture2D<float4> gOutput : BAMBOO_UAV(FRAME_REG_RAYTRACE_OUTPUT);

RaytracingAccelerationStructure SceneBVH : BAMBOO_SRV(FRAME_REG_TLAS);
ByteAddressBuffer g_vertices              : BAMBOO_SRV(FRAME_REG_VERTICES);
ByteAddressBuffer g_indices               : BAMBOO_SRV(FRAME_REG_INDICES);

StructuredBuffer<GeometryInfo>     g_geometryInfo      : BAMBOO_SRV(FRAME_REG_GEOMETRY_INFO);
StructuredBuffer<InstanceInfo>     g_instanceInfo      : BAMBOO_SRV(FRAME_REG_INSTANCE_INFO);
StructuredBuffer<EmissiveTriangle> g_emissiveTriangles : BAMBOO_SRV(FRAME_REG_EMISSIVE_TRIANGLES);
StructuredBuffer<LightData>        g_lightData         : BAMBOO_SRV(FRAME_REG_LIGHT_DATA);
StructuredBuffer<LightPoolEntry>   g_lightPool         : BAMBOO_SRV(FRAME_REG_LIGHT_POOL);

Texture2D g_skybox : BAMBOO_SRV(FRAME_REG_SKYBOX);

// Last, so the bindless array can grow without moving anything above it.
Texture2D g_textures[MAX_TEXTURES] : BAMBOO_SRV(FRAME_REG_MATERIAL_TEXTURES);

SamplerState gsamPointWrap : register(s0);
SamplerState gsamPointClamp : register(s1);
SamplerState gsamLinearWrap : register(s2);
SamplerState gsamLinearClamp : register(s3);
SamplerState gsamAnisotropicWrap : register(s4);
SamplerState gsamAnisotropicClamp : register(s5);

cbuffer CameraParams : BAMBOO_CBV(FRAME_REG_CAMERA_MATRICES)
{
    float4x4 worldViewProj;
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
}

#endif // FRAME_BINDINGS_HLSL
