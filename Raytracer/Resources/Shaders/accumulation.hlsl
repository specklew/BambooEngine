#include "PassRegisters.h"

Texture2D<float4>   gCurrent : BAMBOO_SRV(ACCUM_REG_CURRENT);  // current frame (from m_outputResource)
RWTexture2D<float4> gAccum   : BAMBOO_UAV(ACCUM_REG_ACCUM);    // float running average
RWTexture2D<float4> gDisplay : BAMBOO_UAV(ACCUM_REG_DISPLAY);  // UNORM display output

cbuffer AccumCB : BAMBOO_CBV(ACCUM_REG_CB)
{
    uint frameCount;  // 1 on first frame after reset → weight = 1.0 (full replace)
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 id = tid.xy;
    float3 current = gCurrent[id].rgb;
    float  w       = 1.0 / float(max(frameCount, 1u));
    float3 accum   = lerp(gAccum[id].rgb, current, w);
    gAccum[id]    = float4(accum, 1.0);
    gDisplay[id]  = float4(accum, 1.0);
}
