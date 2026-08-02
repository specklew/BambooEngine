// Geometry-bake clear: resets occupancy and the baked per-voxel bound buffers
// before the bake raster. Bounds start inverted (min = all ones, max = 0) so
// the bake's InterlockedMin/Max shrink-wrap correctly.

#include "PassRegisters.h"

RWTexture3D<uint>        gOccupancy     : BAMBOO_UAV(VOXEL_BAKE_CLEAR_REG_OCCUPANCY);
RWStructuredBuffer<uint> gBakedBoundMin : BAMBOO_UAV(VOXEL_BAKE_CLEAR_REG_BAKED_BOUND_MIN); // 4 uints per cell
RWStructuredBuffer<uint> gBakedBoundMax : BAMBOO_UAV(VOXEL_BAKE_CLEAR_REG_BAKED_BOUND_MAX); // 4 uints per cell

cbuffer ClearCB : BAMBOO_CBV(VOXEL_BAKE_CLEAR_REG_CB)
{
    uint gGridDim;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[numthreads(8, 8, 8)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= gGridDim)) return;

    gOccupancy[tid] = 0u;

    uint flatId = tid.x + tid.y * gGridDim + tid.z * gGridDim * gGridDim;
    [unroll] for (uint i = 0; i < 4; ++i)
    {
        gBakedBoundMin[flatId * 4 + i] = 0xffffffffu;
        gBakedBoundMax[flatId * 4 + i] = 0u;
    }
}
