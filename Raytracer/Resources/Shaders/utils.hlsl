#ifndef UTILS_HLSL
#define UTILS_HLSL

float clamp(float value, float minVal, float maxVal)
{
    return max(minVal, min(value, maxVal));
}

float2 clamp(float2 value, float minVal, float maxVal)
{
    return max(float2(minVal, minVal), min(value, float2(maxVal, maxVal)));
}

float3 clamp(float3 value, float minVal, float maxVal)
{
    return max(float3(minVal, minVal, minVal), min(value, float3(maxVal, maxVal, maxVal)));
}

float4 clamp(float4 value, float minVal, float maxVal)
{
    return max(float4(minVal, minVal, minVal, minVal), min(value, float4(maxVal, maxVal, maxVal, maxVal)));
}

#endif
