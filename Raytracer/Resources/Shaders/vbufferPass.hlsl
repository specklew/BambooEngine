#ifndef VBUFFER_PASS_HLSL
#define VBUFFER_PASS_HLSL

// VBuffer pass (ADR 0004, SIByL raytraced-vbuffer.slang): one jittered primary
// ray per pixel per frame; the hit's identity (instance + primitive +
// barycentrics) is packed into gVBuffer for light injection and the guided
// integrator to reconstruct from — neither traces its own primary anymore.

#include "RaytracingUtils.hlsl"
#include "passConstants.hlsl"
#include "VBuffer.hlsl"

RWTexture2D<uint4> gVBuffer : register(u9);

struct VBufferPayload
{
    uint   primitiveId;
    uint   instanceId;
    float2 barycentrics;
};

[shader("raygeneration")]
void VBufferRayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;

    float3 origin, direction;
    float2 jitter = VBufferPixelJitter(launchIndex, frameIndex, vbufferJitterEnabled);
    GenerateCameraRayJittered(launchIndex, jitter, origin, direction);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    VBufferPayload payload;
    payload.primitiveId = VBUFFER_INVALID;
    payload.instanceId = 0;
    payload.barycentrics = float2(0, 0);
    TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, payload);

    VBufferData data;
    data.primitiveId  = payload.primitiveId;
    data.instanceId   = payload.instanceId;
    data.barycentrics = payload.barycentrics;
    gVBuffer[launchIndex] = PackVBufferData(data);
}

[shader("miss")]
void VBufferMiss(inout VBufferPayload payload : SV_RayPayload)
{
    payload.primitiveId = VBUFFER_INVALID;
}

[shader("anyhit")]
void VBufferAnyHit(inout VBufferPayload payload : SV_RayPayload, in Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    float4 albedo = SampleTextureColor(hit) * instance.baseColorFactor;
    if (albedo.a < 0.001)
    {
        IgnoreHit();
    }
}

[shader("closesthit")]
void VBufferHit(inout VBufferPayload payload : SV_RayPayload, in Attributes attr)
{
    payload.primitiveId  = PrimitiveIndex();
    payload.instanceId   = InstanceID();
    payload.barycentrics = attr.barycentrics;
}

#endif
