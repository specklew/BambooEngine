#include "PassRegisters.h"

Texture2D<float4>   gCurrent : BAMBOO_PASS_SRV(ACCUM_REG_CURRENT);  // current frame (from m_outputResource)
RWTexture2D<float4> gAccum   : BAMBOO_PASS_UAV(ACCUM_REG_ACCUM);    // float running average
RWTexture2D<float4> gDisplay : BAMBOO_PASS_UAV(ACCUM_REG_DISPLAY);  // UNORM display output

// xyz = Welford M2 per channel, w = the running mean's luminance (see PassRegisters.h).
RWStructuredBuffer<float4> gVarianceM2 : BAMBOO_PASS_UAV(ACCUM_REG_VARIANCE_M2);

cbuffer AccumCB : BAMBOO_PASS_CBV(ACCUM_REG_CB)
{
    uint frameCount;      // 1 on first frame after reset → weight = 1.0 (full replace)
    uint varianceEnabled; // renderer.accumulation.variance
    uint imageWidth;      // the buffer write needs bounds; the texture writes do not
    uint imageHeight;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 id = tid.xy;
    float3 current = gCurrent[id].rgb;
    float3 previous = gAccum[id].rgb;
    float  w       = 1.0 / float(max(frameCount, 1u));
    float3 accum   = lerp(previous, current, w);
    gAccum[id]    = float4(accum, 1.0);
    gDisplay[id]  = float4(accum, 1.0);

    if (varianceEnabled == 0u || id.x >= imageWidth || id.y >= imageHeight)
        return;

    // Welford: M2 += (x - mean_before) * (x - mean_after). The lerp above IS the
    // running mean, so both terms are already here. The first sample must not read
    // the previous mean at all — the buffer is undefined before a reset, and
    // garbage * 0 is NaN, not 0.
    const uint index = id.y * imageWidth + id.x;
    float3 m2 = float3(0, 0, 0);
    if (frameCount > 1u)
        m2 = gVarianceM2[index].rgb + (current - previous) * (current - accum);

    gVarianceM2[index] = float4(m2, dot(accum, float3(0.2126, 0.7152, 0.0722)));
}
