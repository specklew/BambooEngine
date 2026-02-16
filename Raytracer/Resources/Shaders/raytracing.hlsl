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
    uint textureIndex;
};

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

ByteAddressBuffer g_vertices : register(t1);
ByteAddressBuffer g_indices : register(t2);

StructuredBuffer<GeometryInfo> g_geometryInfo : register(t3);
StructuredBuffer<InstanceInfo> g_instanceInfo : register(t4);

Texture2D g_textures[MAX_TEXTURES] : register(t5);

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

uint3 Load3x32BitIndices(uint triangleIndex)
{
    uint3 indices;
    indices.x = g_indices.Load((triangleIndex * 3 + 0) * 4);
    indices.y = g_indices.Load((triangleIndex * 3 + 1) * 4);
    indices.z = g_indices.Load((triangleIndex * 3 + 2) * 4);
    
    return indices;
}

float2 GetUVAttribute(uint byteOffset)
{
    return asfloat(g_vertices.Load2(byteOffset));
}

[shader("raygeneration")] 
void RayGen() {
    // Initialize the ray payload
    HitInfo payload = {float4(0, 0, 0, 0)};

    // Get the location within the dispatched 2D grid of work items
    // (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;

    float2 dims = float2(DispatchRaysDimensions().xy);
    float2 d = (((launchIndex.xy + 0.5f) / dims) * 2.f - 1.f);

    RayDesc ray;
    ray.Origin = mul(viewI, float4(0, 0, 0, 1));
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    ray.Direction = mul(viewI, float4(target.xyz, 0));
    ray.TMin = 0.01;
    ray.TMax = 1000;

    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    float3 color = payload.colorAndDistance.xyz; 
    gOutput[launchIndex] = float4(color, 1.f);
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload)
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);
    
    float ramp = launchIndex.y / dims.y;
    payload.colorAndDistance = float4(0.3f, 0.4f, 0.7f - 0.2f*ramp, -1.0f);
}

struct STriVertex
{
    float3 vertex;
    //float4 color;
};

[shader("closesthit")] 
void Hit(inout HitInfo payload : SV_RayPayload, Attributes attr) 
{
    uint geometryIndex = g_instanceInfo[InstanceIndex()].geometryIndex;
    uint vertexOffset = g_geometryInfo[geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[geometryIndex].indexOffset / 3;
    
    uint3 indices = Load3x32BitIndices(PrimitiveIndex() + indexOffset);
    indices.x += vertexOffset;
    indices.y += vertexOffset;
    indices.z += vertexOffset;
    float2 uv0 = GetUVAttribute(indices.x * 32 + 24);
    float2 uv1 = GetUVAttribute(indices.y * 32 + 24);
    float2 uv2 = GetUVAttribute(indices.z * 32 + 24);

    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    
    float4 col = float4(1, 1, 1, 1);

    if (g_instanceInfo[InstanceIndex()].textureIndex != -1)
    {
        col = g_textures[g_instanceInfo[InstanceIndex()].textureIndex].SampleLevel(gsamLinearWrap, uv, 0);
    }
    
    payload.colorAndDistance = float4(col.x, col.y, col.z, RayTCurrent()); 
}
