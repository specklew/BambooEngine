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

uint3 Load3x32BitIndices(uint triangleIndex, uint indexOffset = 0)
{
    // byte to load = ((instance triangle) * 3 + (vert in triangle) + offset) * (bytes per index)
    uint3 indices;
    indices.x = g_indices.Load((triangleIndex * 3 + 0 + indexOffset) * 4);
    indices.y = g_indices.Load((triangleIndex * 3 + 1 + indexOffset) * 4);
    indices.z = g_indices.Load((triangleIndex * 3 + 2 + indexOffset) * 4);
    
    return indices;
}

float2 GetUVAttribute(uint byteOffset)
{
    return asfloat(g_vertices.Load2(byteOffset));
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

    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

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
    uint geometryIndex = g_instanceInfo[InstanceID()].geometryIndex;
    uint vertexOffset = g_geometryInfo[geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[geometryIndex].indexOffset;
    
    uint3 indices = Load3x32BitIndices(PrimitiveIndex(), indexOffset);
    indices += vertexOffset;
    
    float2 uv0 = GetUVAttribute(indices.x * 32 + 24);
    float2 uv1 = GetUVAttribute(indices.y * 32 + 24);
    float2 uv2 = GetUVAttribute(indices.z * 32 + 24);

    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    // ---------------------
    // Calculate UV gradient
    // ---------------------

    // Triangle normal
    float3 p0 = asfloat(g_vertices.Load3(indices.x * 32));
    float3 p1 = asfloat(g_vertices.Load3(indices.y * 32));
    float3 p2 = asfloat(g_vertices.Load3(indices.z * 32));
    float3 triangleNormal = normalize(cross(p2 - p0, p1 - p0));
    
    uint2 launchIndex = DispatchRaysIndex().xy;
    float3 ddxOrigin, ddxDirection, ddyOrigin, ddyDirection;
    GenerateCameraRay(launchIndex + uint2(1, 0), ddxOrigin, ddxDirection);
    GenerateCameraRay(launchIndex + uint2(0, 1), ddyOrigin, ddyDirection);

    float3 xOffsetPoint = RayPlaneIntersection(p0, triangleNormal, ddxOrigin, ddxDirection);
    float3 yOffsetPoint = RayPlaneIntersection(p0, triangleNormal, ddyOrigin, ddyDirection);

    float3 baryX = BarycentricCoordinates(xOffsetPoint, p0, p1, p2);
    float3 baryY = BarycentricCoordinates(yOffsetPoint, p0, p1, p2);

    float3x2 uvMat = float3x2(uv0, uv1, uv2);
    float2 ddxUV = mul(baryX, uvMat) - uv;
    float2 ddyUV = mul(baryY, uvMat) - uv;
    
    float4 col = float4(1, 0, 1, 1);

    //col = g_textures[4].Load(0) / 1.0f;
    
    if (g_instanceInfo[InstanceID()].textureIndex != -1)
    {
        //uv = float2(0.5, 0.5);
        //col = g_textures[g_instanceInfo[InstanceID()].textureIndex].Load(0);
        col = g_textures[InstanceID()].Load(0);
    }
    
    payload.colorAndDistance = float4(col.x, col.y, col.z, RayTCurrent()); 
}
