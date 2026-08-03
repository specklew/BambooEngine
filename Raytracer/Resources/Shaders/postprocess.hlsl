#include "PassRegisters.h"

Texture2D<float4>   gInput  : BAMBOO_PASS_SRV(POST_REG_INPUT);
RWTexture2D<float4> gOutput : BAMBOO_PASS_UAV(POST_REG_OUTPUT);

cbuffer PostProcessCB : BAMBOO_PASS_CBV(POST_REG_CB)
{
    float exposure;
    float contrast;    // pre-ACES power curve  (default 1.0)
    float saturation;  // post-ACES luma lerp   (default 1.0)
    float lift;        // post-ACES shadow lift  (default 0.0)
};

// ACES filmic tonemap (Narkowicz 2015 approximation)
float3 ACESFilmic(float3 x)
{
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 id = tid.xy;
    float3 color = gInput[id].rgb * exposure;

    // Pre-ACES contrast (power curve)
    color = pow(max(color, 0.0), contrast);

    // ACES filmic
    color = ACESFilmic(color);

    // Post-ACES shadow lift
    color += lift * (1.0 - color);

    // Post-ACES saturation (Rec.709 luma weights)
    float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    color = lerp(luma.xxx, color, saturation);

    color = pow(color, 1.0 / 2.2);  // linear → gamma (sRGB approximate)
    gOutput[id] = float4(color, 1.0);
}
