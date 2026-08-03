// Per-frame clear of the injection accumulators. Occupancy is NOT cleared —
// it is a bake output that persists until the next rebake (ADR 0004).

#include "PassRegisters.h"

RWTexture3D<uint> gIrradiance : BAMBOO_PASS_UAV(VOXEL_FRAME_CLEAR_REG_IRRADIANCE);
RWTexture3D<uint> gVplCount   : BAMBOO_PASS_UAV(VOXEL_FRAME_CLEAR_REG_VPL_COUNT);

cbuffer ClearCB : BAMBOO_PASS_CBV(VOXEL_FRAME_CLEAR_REG_CB)
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

    gIrradiance[tid] = 0u;
    gVplCount[tid]   = 0u;
}
