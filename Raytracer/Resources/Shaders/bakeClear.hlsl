// Geometry-bake clear: resets occupancy and the baked per-voxel bound buffers
// before the bake raster. Bounds start inverted (min = all ones, max = 0) so
// the bake's InterlockedMin/Max shrink-wrap correctly.

RWTexture3D<uint>        gOccupancy     : register(u0);
RWStructuredBuffer<uint> gBakedBoundMin : register(u1); // 4 uints per cell
RWStructuredBuffer<uint> gBakedBoundMax : register(u2); // 4 uints per cell

cbuffer ClearCB : register(b0)
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
