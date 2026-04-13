#ifndef RANDOM_HLSL
#define RANDOM_HLSL

// PCG hash — high quality, cheap
uint pcg_hash(uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Returns float2 in [0, 1)
float2 Random2D(uint seed)
{
    return float2(
        float(pcg_hash(seed))                * (1.0 / 4294967296.0),
        float(pcg_hash(seed ^ 0x9e3779b9u))  * (1.0 / 4294967296.0)
    );
}

#endif // RANDOM_HLSL