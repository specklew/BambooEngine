// Hit information, aka ray payload
// This sample only carries a shading color and hit distance.
// Note that the payload should be kept as small as possible,
// and that its size must be declared in the corresponding
// D3D12_RAYTRACING_SHADER_CONFIG pipeline subobjet.
struct HitInfo
{
    float4 colorAndDistance;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes
{
    float2 bary;
};

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

cbuffer CameraParams : register(b0)
{
    float4x4 worldViewProj;
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
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
    ray.TMin = 0.001;
    ray.TMax = 1000;

    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
  
    gOutput[launchIndex] = payload.colorAndDistance;
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
void Hit(inout HitInfo payload : SV_RayPayload, Attributes attrib) 
{
    float3 barycentrics = 
    float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);
    
    //uint vertId = 3 * PrimitiveIndex();
    //float3 hitColor = BTriVertex[indices[vertId + 0]].color * barycentrics.x +
    //BTriVertex[indices[vertId + 1]].color * barycentrics.y +
    //BTriVertex[indices[vertId + 2]].color * barycentrics.z;
  
    payload.colorAndDistance = float4(barycentrics.x,barycentrics.y,barycentrics.z, 0); 
}
