// VXPG guiding distribution build: reload baked geometry bounds for lit voxels,
// then compact nonzero-irradiance voxels into a flat list. Entry points
// dispatched in order each frame after light injection:
//   ClearCounters -> BakeReload -> CompactVoxels
// (SIByL vxguiding: bake-reload.slang dense branch + geometry/compact.slang.
// The V1 flat-CDF kernel is gone — voxel selection lives in the light tree.)

#define VOXEL_GUIDING_CAPACITY 131072

RWTexture3D<uint>   gVoxIrradiance        : register(u0);
RWTexture3D<uint>   gVoxVplCount          : register(u1);
// Representative VPL surface position + octahedral normal per voxel, written
// during light injection (last-writer-wins).
RWTexture3D<float4> gVoxelRepresentative  : register(u2);

// [0] = compacted voxel count ([1] retired with the flat CDF)
RWStructuredBuffer<uint>   gCounters     : register(u3);
RWStructuredBuffer<uint>   gCompactIds   : register(u4);
// voxelID (flat) -> compactID, sentinel -1. Sized gridDim^3. Each cell written
// only by its own CompactVoxels thread, so it doubles as its own clear.
RWStructuredBuffer<int>    gInverseIndex : register(u5);
// Live per-voxel bounds, quantized to the voxel cube. BakeReload copies the
// baked values in for lit voxels; unlit cells are stale and never read.
RWStructuredBuffer<uint4>  gLiveBoundMin : register(u6);
RWStructuredBuffer<uint4>  gLiveBoundMax : register(u7);
// Compact-indexed outputs for the guiding passes downstream (fingerprint, tree).
// SIByL u_RepresentVPL. One representative light-carrying surface point per
// compact voxel (pos + octa normal), consumed by the fingerprint shadow rays.
RWStructuredBuffer<float4> gCompactVoxelLightPoints : register(u8);
RWStructuredBuffer<float>  gPremulIrradiance  : register(u9);

// Bake outputs (read-only here; UAV-typed to stay in UNORDERED_ACCESS and skip
// per-bake state transitions).
RWStructuredBuffer<uint4> gBakedBoundMin : register(u10);
RWStructuredBuffer<uint4> gBakedBoundMax : register(u11);

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
    gInverseIndex[flatId] = int(slot);

    // Zero representative = "no VPL landed here" — written through so the
    // consumer sees an explicit sentinel instead of stale data.
    gCompactVoxelLightPoints[slot] = gVoxelRepresentative[tid];

    // Irradiance premultiplied by the bound's dominant-face area. Full-cube
    // bounds (bake flags off) give area = 1, so this equals the plain weight;
    // an unbaked-but-lit cell unpacks inverted (extend <= 0) and premuls to 0.
    const float maxUint = 4294967295.0f;
    float3 boundMin = float3(gLiveBoundMin[flatId].xyz) / maxUint;
    float3 boundMax = float3(gLiveBoundMax[flatId].xyz) / maxUint;
    float3 extend = max(boundMax - boundMin, 0.0f);
    gPremulIrradiance[slot] = weight * DominantFaceArea(extend);
}
