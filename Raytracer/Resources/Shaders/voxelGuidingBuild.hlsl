// VXPG guiding distribution build: reload baked geometry bounds for lit voxels,
// compact nonzero-irradiance voxels into a flat list, then build an
// inclusive-prefix-sum CDF for binary-search sampling. Entry points dispatched
// in order each frame after light injection:
//   ClearCounters -> BakeReload -> CompactVoxels -> BuildCdf
// (SIByL vxguiding: bake-reload.slang dense branch + geometry/compact.slang.)

#define VOXEL_GUIDING_CAPACITY 131072

RWTexture3D<uint>   gVoxIrradiance        : register(u0);
RWTexture3D<uint>   gVoxVplCount          : register(u1);
// Representative VPL surface position + octahedral normal per voxel, written
// during light injection (last-writer-wins).
RWTexture3D<float4> gVoxelRepresentative  : register(u2);

// [0] = compacted voxel count, [1] = asuint(total weight)
RWStructuredBuffer<uint>   gCounters     : register(u3);
RWStructuredBuffer<uint>   gCompactIds   : register(u4);
RWStructuredBuffer<float>  gWeights      : register(u5);
RWStructuredBuffer<float>  gCdf          : register(u6);
// voxelID (flat) -> compactID, sentinel -1. Sized gridDim^3. Each cell written
// only by its own CompactVoxels thread, so it doubles as its own clear.
RWStructuredBuffer<int>    gInverseIndex : register(u7);
// Live per-voxel bounds, quantized to the voxel cube. BakeReload copies the
// baked values in for lit voxels; unlit cells are stale and never read.
RWStructuredBuffer<uint4>  gLiveBoundMin : register(u8);
RWStructuredBuffer<uint4>  gLiveBoundMax : register(u9);
// Compact-indexed outputs for the guiding passes downstream (fingerprint, tree).
RWStructuredBuffer<float4> gRepresentVPL      : register(u10);
RWStructuredBuffer<float>  gPremulIrradiance  : register(u11);

// Bake outputs (read-only here; UAV-typed to stay in UNORDERED_ACCESS and skip
// per-bake state transitions).
RWStructuredBuffer<uint4> gBakedBoundMin : register(u12);
RWStructuredBuffer<uint4> gBakedBoundMax : register(u13);

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

// Copy baked per-voxel bounds into the live buffers for voxels that received
// irradiance this frame (bake-reload.slang, dense/Unroll branch, gated on the
// raw packed irradiance being nonzero).
[numthreads(8, 8, 8)]
void BakeReload(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= gGridDim)) return;

    if (gVoxIrradiance[tid] == 0u) return;

    uint flatId = tid.x + tid.y * gGridDim + tid.z * gGridDim * gGridDim;
    gLiveBoundMin[flatId] = gBakedBoundMin[flatId];
    gLiveBoundMax[flatId] = gBakedBoundMax[flatId];
}

// Dominant-face area of a bound normalized to the unit voxel cube
// (compact.slang): the face spanned by the two largest extents.
float DominantFaceArea(float3 extend)
{
    if (extend.x <= extend.y && extend.x <= extend.z) return extend.y * extend.z;
    if (extend.y <= extend.x && extend.y <= extend.z) return extend.x * extend.z;
    return extend.x * extend.y;
}

[numthreads(8, 8, 8)]
void CompactVoxels(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= gGridDim)) return;

    uint flatId = tid.x + tid.y * gGridDim + tid.z * gGridDim * gGridDim;
    gInverseIndex[flatId] = -1; // clear own cell before any early-return

    uint count = gVoxVplCount[tid];
    if (count == 0u) return;

    float weight = UnpackIrradiance(gVoxIrradiance[tid]) / float(count);
    if (weight <= 0.0f) return;

    uint slot;
    InterlockedAdd(gCounters[0], 1u, slot);
    if (slot >= VOXEL_GUIDING_CAPACITY) return; // overflow: drop voxel

    gCompactIds[slot] = flatId;
    gWeights[slot] = weight;
    gInverseIndex[flatId] = int(slot);

    // Zero representative = "no VPL landed here" — written through so the
    // consumer sees an explicit sentinel instead of stale data.
    gRepresentVPL[slot] = gVoxelRepresentative[tid];

    // Irradiance premultiplied by the bound's dominant-face area. Full-cube
    // bounds (bake flags off) give area = 1, so this equals the plain weight;
    // an unbaked-but-lit cell unpacks inverted (extend <= 0) and premuls to 0.
    const float maxUint = 4294967295.0f;
    float3 boundMin = float3(gLiveBoundMin[flatId].xyz) / maxUint;
    float3 boundMax = float3(gLiveBoundMax[flatId].xyz) / maxUint;
    float3 extend = max(boundMax - boundMin, 0.0f);
    gPremulIrradiance[slot] = weight * DominantFaceArea(extend);
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
