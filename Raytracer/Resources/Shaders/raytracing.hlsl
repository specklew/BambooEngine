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
    float3 n0;
    float3 n1;
    float3 n2;
    float3 normal;
    float3 v0;
    float3 v1;
    float3 v2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
};

struct HitData
{
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

float2 GetVertexFloat2Attribute(uint byteOffset)
{
    return asfloat(g_vertices.Load2(byteOffset));
}

float3 GetVertexFloat3Attribute(uint byteOffset)
{
    return asfloat(g_vertices.Load3(byteOffset));
}

TriangleData GetTriangleData(uint triangleIndex, uint vertexOffset, uint indexOffset)
{
    TriangleData data;
    
    data.indices = Load3x32BitIndices(triangleIndex, indexOffset) + vertexOffset;

    data.uv0 = GetVertexFloat2Attribute(data.indices.x * 32 + 24);
    data.uv1 = GetVertexFloat2Attribute(data.indices.y * 32 + 24);
    data.uv2 = GetVertexFloat2Attribute(data.indices.z * 32 + 24);

    data.v0 = GetVertexFloat3Attribute(data.indices.x * 32 + 0);
    data.v1 = GetVertexFloat3Attribute(data.indices.y * 32 + 0);
    data.v2 = GetVertexFloat3Attribute(data.indices.z * 32 + 0);

    data.n0 = GetVertexFloat3Attribute(data.indices.x * 32 + 12);
    data.n1 = GetVertexFloat3Attribute(data.indices.y * 32 + 12);
    data.n2 = GetVertexFloat3Attribute(data.indices.z * 32 + 12);

    data.normal = normalize(cross(data.v2 - data.v0, data.v1 - data.v0));
    
    return data;
}

HitData GetHitData(TriangleData triangleData, float2 barycentrics)
{
    HitData data;
    
    float3 bary = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    
    data.uv = bary.x * triangleData.uv0 + bary.y * triangleData.uv1 + bary.z * triangleData.uv2;
    data.normal = normalize(bary.x * triangleData.n0 + bary.y * triangleData.n1 + bary.z * triangleData.n2);
    data.position = bary.x * triangleData.v0 + bary.y * triangleData.v1 + bary.z * triangleData.v2;
    
    return data;
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
    
    TriangleData triangle_data = GetTriangleData(PrimitiveIndex(), vertexOffset, indexOffset);
    HitData hit_data = GetHitData(triangle_data, attr.barycentrics);

    if (payload.colorAndDistance.z > 0.0f || InstanceID() != 4) // already hit something closer, so ignore this hit
    {
        // ---------------------
        // Calculate UV gradient
        // ---------------------

        // Triangle normal
        float3 triangleNormal = triangle_data.normal;
    
        uint2 launchIndex = DispatchRaysIndex().xy;
        float3 ddxOrigin, ddxDirection, ddyOrigin, ddyDirection;
        GenerateCameraRay(launchIndex + uint2(1, 0), ddxOrigin, ddxDirection);
        GenerateCameraRay(launchIndex + uint2(0, 1), ddyOrigin, ddyDirection);

        float3 xOffsetPoint = RayPlaneIntersection(triangle_data.v0, triangleNormal, ddxOrigin, ddxDirection);
        float3 yOffsetPoint = RayPlaneIntersection(triangle_data.v0, triangleNormal, ddyOrigin, ddyDirection);

        float3 baryX = BarycentricCoordinates(xOffsetPoint, triangle_data.v0, triangle_data.v1, triangle_data.v2);
        float3 baryY = BarycentricCoordinates(yOffsetPoint, triangle_data.v0, triangle_data.v1, triangle_data.v2);

        float3x2 uvMat = float3x2(triangle_data.uv0, triangle_data.uv1, triangle_data.uv2);
        float2 ddxUV = mul(baryX, uvMat) - hit_data.uv;
        float2 ddyUV = mul(baryY, uvMat) - hit_data.uv;
    
        float4 col = float4(1, 0, 1, 1);
    
        if (g_instanceInfo[InstanceID()].textureIndex != -1)
        {
            col = g_textures[g_instanceInfo[InstanceID()].textureIndex].SampleGrad(gsamAnisotropicWrap, hit_data.uv, ddxUV, ddyUV);
        }
    
        payload.colorAndDistance = float4(col.x, col.y, col.z, RayTCurrent()); 
        return;
    }

    payload.colorAndDistance.z = 1.0f;
    
    RayDesc ray;
    ray.Origin = hit_data.position;
    ray.Direction = reflect(WorldRayDirection(), hit_data.normal);
    ray.TMin = 0.01;
    ray.TMax = 1000;

    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
}
