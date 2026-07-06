#ifndef PASS_CONSTANTS_HLSL
#define PASS_CONSTANTS_HLSL

cbuffer PassConstants : register(b3)
{
    float uvX;
    float uvY;
    int debugMode;
    uint numLights;
    uint samplesPerPixel;
    uint numBounces;
    uint frameIndex;
    uint guidingFlags; // bit 0 = power MIS heuristic (also fills the implicit pad before float3)
    float3 cameraWorldPos;
    // 1 = apply per-pixel sub-pixel jitter to the shared VBuffer primaries; the
    // jitter itself is derived per pixel from (pixel, frameIndex) in-shader.
    uint vbufferJitterEnabled;
    uint _passPad0;
}

#endif
