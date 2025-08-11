#include "Common.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

cbuffer CameraParams : register(b0)
{
  float4x4 view;
  float4x4 projection;
  float4x4 viewI;
  float4x4 projectionI;
}


[shader("raygeneration")] 
void RayGen() {
  // Initialize the ray payload
  HitInfo payload;
  payload.colorAndDistance = float4(0, 0, 0, 0);

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
  ray.TMax = 100000;

  TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
  
  gOutput[launchIndex] = payload.colorAndDistance;
}
