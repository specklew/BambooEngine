#ifndef VOXEL_GRID_HLSLI
#define VOXEL_GRID_HLSLI

struct VoxelGridParams
{
    float3 gridMin;
    float  voxelSize;
    float3 gridMax;
    uint   gridDim;
};

int3 WorldToVoxelIndex(VoxelGridParams p, float3 wpos)
{
    return int3(floor((wpos - p.gridMin) / p.voxelSize));
}

bool VoxelInBounds(VoxelGridParams p, int3 idx)
{
    return all(idx >= 0) && all(idx < int(p.gridDim));
}

float3 WorldToNdc(VoxelGridParams p, float3 wpos)
{
    float3 t = (wpos - p.gridMin) / (p.gridMax - p.gridMin);
    return t * 2.0f - 1.0f;
}

float4 ProjectForAxis(VoxelGridParams p, float3 wpos, uint axisIndex)
{
    float3 ndc = WorldToNdc(p, wpos);
    // Each axis projects orthographically onto its perpendicular plane.
    // SV_Position xy = plane coords, z = 0.5 (depth not used; depth test disabled).
    float2 xy;
    if (axisIndex == 0)      xy = float2(ndc.y, ndc.z); // looking down +X
    else if (axisIndex == 1) xy = float2(ndc.x, ndc.z); // looking down +Y
    else                     xy = float2(ndc.x, ndc.y); // looking down +Z
    return float4(xy, 0.5f, 1.0f);
}

#endif
