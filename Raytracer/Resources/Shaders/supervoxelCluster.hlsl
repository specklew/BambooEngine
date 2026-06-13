// VXPG V2 Stage A: supervoxel clustering.
// Supervoxels are coarse grid cells (voxelCoord / clusterFactor). This pass
// accumulates, per supervoxel, the summed scalar irradiance of its active
// voxels and a count of active voxels. The supervoxel AABB/centroid is analytic
// from its grid index (see GuidingMatrixBuildPass), so nothing spatial is stored
// here. Two entry points dispatched in order each frame after the guiding build:
//   ClearSupervoxels -> AccumulateSupervoxels

#define MAX_SUPERVOXELS 512

RWTexture3D<uint> gVoxIrradiance : register(u0);
RWTexture3D<uint> gVoxVplCount   : register(u1);

// gSvIrradiance: packed fixed-point (x100) summed irradiance per supervoxel.
// gSvCount:      active voxel count per supervoxel (0 => supervoxel inactive).
RWStructuredBuffer<uint> gSvIrradiance : register(u2);
RWStructuredBuffer<uint> gSvCount      : register(u3);

cbuffer SupervoxelCB : register(b0)
{
    uint gGridDim;
    uint gClusterFactor;
    uint gSvDim;       // ceil(gridDim / clusterFactor)
    uint _pad0;
}

[numthreads(256, 1, 1)]
void ClearSupervoxels(uint tid : SV_DispatchThreadID)
{
    if (tid >= MAX_SUPERVOXELS) return;
    gSvIrradiance[tid] = 0u;
    gSvCount[tid] = 0u;
}

[numthreads(8, 8, 8)]
void AccumulateSupervoxels(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= gGridDim)) return;

    uint count = gVoxVplCount[tid];
    if (count == 0u) return;

    uint packedIrr = gVoxIrradiance[tid];
    if (packedIrr == 0u) return;

    // Per-voxel averaged irradiance, repacked to fixed-point for the atomic sum.
    float irr = (float(packedIrr) / 100.0f) / float(count);
    uint contribution = uint(irr * 100.0f);
    if (contribution == 0u) return;

    uint3 sv = tid / gClusterFactor;
    uint svIndex = sv.x + sv.y * gSvDim + sv.z * gSvDim * gSvDim;
    if (svIndex >= MAX_SUPERVOXELS) return; // overflow guard

    InterlockedAdd(gSvIrradiance[svIndex], contribution);
    InterlockedAdd(gSvCount[svIndex], 1u);
}
