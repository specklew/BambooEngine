// VXPG guiding distribution build: compact nonzero-irradiance voxels into a
// flat list, then build an inclusive-prefix-sum CDF for binary-search sampling.
// Three entry points dispatched in order each frame after light injection:
//   ClearCounters -> CompactVoxels -> BuildCdf

#define VOXEL_GUIDING_CAPACITY 131072

RWTexture3D<uint> gVoxIrradiance : register(u0);
RWTexture3D<uint> gVoxVplCount   : register(u1);

// [0] = compacted voxel count, [1] = asuint(total weight)
RWStructuredBuffer<uint>  gCounters   : register(u2);
RWStructuredBuffer<uint>  gCompactIds : register(u3);
RWStructuredBuffer<float> gWeights    : register(u4);
RWStructuredBuffer<float> gCdf        : register(u5);

cbuffer BuildCB : register(b0)
{
    uint gGridDim;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

float UnpackIrradiance(uint packed)
{
    return float(packed) / 100.0f;
}

[numthreads(1, 1, 1)]
void ClearCounters(uint3 tid : SV_DispatchThreadID)
{
    gCounters[0] = 0u;
    gCounters[1] = asuint(0.0f);
}

[numthreads(8, 8, 8)]
void CompactVoxels(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= gGridDim)) return;

    uint count = gVoxVplCount[tid];
    if (count == 0u) return;

    float weight = UnpackIrradiance(gVoxIrradiance[tid]) / float(count);
    if (weight <= 0.0f) return;

    uint slot;
    InterlockedAdd(gCounters[0], 1u, slot);
    if (slot >= VOXEL_GUIDING_CAPACITY) return; // overflow: drop voxel

    gCompactIds[slot] = tid.x + tid.y * gGridDim + tid.z * gGridDim * gGridDim;
    gWeights[slot] = weight;
}

// Single-group scan: each thread serially scans its chunk, thread 0 scans the
// 1024 partials, then chunk offsets are added back. n <= capacity (131072)
// gives chunks of <= 128 elements per thread.
groupshared float sPartials[1024];

[numthreads(1024, 1, 1)]
void BuildCdf(uint tid : SV_GroupThreadID)
{
    const uint n = min(gCounters[0], VOXEL_GUIDING_CAPACITY);
    const uint chunk = (n + 1023u) / 1024u;
    const uint begin = tid * chunk;
    const uint end = min(begin + chunk, n);

    float sum = 0.0f;
    for (uint i = begin; i < end; ++i)
    {
        sum += gWeights[i];
        gCdf[i] = sum;
    }
    sPartials[tid] = sum;

    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        float running = 0.0f;
        for (uint p = 0; p < 1024; ++p)
        {
            float v = sPartials[p];
            sPartials[p] = running; // exclusive
            running += v;
        }
        gCounters[1] = asuint(running); // total weight
    }

    GroupMemoryBarrierWithGroupSync();

    const float offset = sPartials[tid];
    if (offset != 0.0f)
    {
        for (uint i = begin; i < end; ++i)
            gCdf[i] += offset;
    }
}
