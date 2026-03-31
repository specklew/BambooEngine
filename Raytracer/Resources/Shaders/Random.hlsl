#ifndef RANDOM_HLSL
#define RANDOM_HLSL

StructuredBuffer<float> g_random : register(t5);

cbuffer Constants : register(b1)
{
    float time;
}

float2 Random2D(float seed)
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint idx = pixel.x + pixel.y * DispatchRaysDimensions().y;
    float base = g_random.Load(idx) + seed + time;

    float r1 = frac(sin(base) * 43758.5453);
    float r2 = frac(cos(base + 1.0) * 22578.1459);
    return float2(abs(r1), abs(r2));
}

#endif // RANDOM_HLSL