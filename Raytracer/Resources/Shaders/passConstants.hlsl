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
    float3 cameraWorldPos;
}

#endif
