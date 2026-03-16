#ifndef RANDOM_HLSL
#define RANDOM_HLSL

// Resource declarations
StructuredBuffer<float> g_random : register(t5);

cbuffer Constants : register(b1)
{
    float time;
}

float rand_1_05(in float2 uv)
{
    float2 noise = (frac(sin(dot(uv, float2(12.9898, 78.233) * 2.0)) * 43758.5453));
    return abs(noise.x + noise.y) * 0.5;
}

float3 RandomDirectionInHemisphere(float3 normal, float seed = 1.0f)
{
    float uv_1 = sin(g_random.Load(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().y) + seed + time);
    float uv_2 = cos(g_random.Load(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().y) + seed + time);
    float2 s1 = float2(uv_1, uv_2);
    float2 s2 = float2(cos(uv_1 + 1), sin(uv_2 + 2));
    float2 s3 = float2(sin(uv_2 + 1), cos(uv_1 + 2));
    float3 randomDirection = float3(rand_1_05(s1), rand_1_05(s2), rand_1_05(s3)) * 2.0f - 1.0f;
    if (dot(randomDirection, normal) < 0.0f)
        randomDirection = -randomDirection;
    return normalize(randomDirection) + normalize(normal);
}

float2 Random2D(float seed)
{
    float r1 = frac(sin(g_random.Load(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().y) + seed + time) * 43758.5453);
    float r2 = frac(cos(g_random.Load(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().y) + seed + time + 1.0) * 22578.1459);
    return float2(abs(r1), abs(r2));
}

#endif // RANDOM_HLSL