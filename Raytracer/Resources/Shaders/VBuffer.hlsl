#ifndef VBUFFER_HLSL
#define VBUFFER_HLSL

// Shared visibility buffer (ADR 0004, SIByL vbuffer.hlsli): one primary hit
// per pixel per frame, stored as IDENTITY (which triangle + where on it), not
// attributes. 128 bits/pixel, RGBA32_UINT:
//   x = primitiveID (0xFFFFFFFF = miss)
//   y = instanceID (indexes g_instanceInfo)
//   z, w = barycentrics as uint bits
// Consumers reconstruct position/normal/material via
// GetHitData(..., instance.objectToWorld).

struct VBufferData
{
    uint   primitiveId;
    uint   instanceId;
    float2 barycentrics;
};

static const uint VBUFFER_INVALID = 0xFFFFFFFFu;

bool IsVBufferInvalid(VBufferData data)
{
    return data.primitiveId == VBUFFER_INVALID;
}

uint4 PackVBufferData(VBufferData data)
{
    return uint4(data.primitiveId, data.instanceId,
                 asuint(data.barycentrics.x), asuint(data.barycentrics.y));
}

VBufferData UnpackVBufferData(uint4 packed)
{
    VBufferData data;
    data.primitiveId    = packed.x;
    data.instanceId     = packed.y;
    data.barycentrics.x = asfloat(packed.z);
    data.barycentrics.y = asfloat(packed.w);
    return data;
}

#endif
