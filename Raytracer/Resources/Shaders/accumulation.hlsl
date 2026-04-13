Texture2D<float4>   gCurrent : register(t0);  // current frame (from m_outputResource)
RWTexture2D<float4> gAccum   : register(u0);  // float running average
RWTexture2D<float4> gDisplay : register(u1);  // UNORM display output

cbuffer AccumCB : register(b0)
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
    gDisplay[id]  = float4(saturate(accum), 1.0);
}
