// Paints one buffer-debug view straight from the VXPG products (ADR 0017 phase
// 5b). The raster-only versions of these views read the rasterized surface
// position; this reads the ShadingPoints G-buffer instead, which is the shared
// VBuffer's primary hit — same content, and it means the view no longer depends
// on which technique is drawing the frame.

#include "PassRegisters.h"
#include "BufferDebugView.h"
#include "Octahedral.hlsl"

RWTexture2D<float4> gDebugOutput    : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_OUTPUT);
RWTexture2D<float4> gShadingPoints  : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_SHADING_POINTS);
RWTexture3D<uint>   gVoxelOccupancy : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_VOXEL_OCCUPANCY);
RWTexture3D<uint>   gVoxelIrradiance: BAMBOO_PASS_UAV(DEBUG_VIEW_REG_VOXEL_IRRADIANCE);
RWTexture3D<uint>   gVoxelVplCount  : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_VOXEL_VPL_COUNT);
RWTexture2D<int>    gSuperpixelIndex : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_SUPERPIXEL_INDEX);
RWTexture2D<float4> gSuperpixelCenter: BAMBOO_PASS_UAV(DEBUG_VIEW_REG_SUPERPIXEL_CENTER);

// Root UAVs, matching how the guided technique binds the same buffers.
RWStructuredBuffer<int>  gVoxInverseIndex         : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_INVERSE_INDEX);
RWStructuredBuffer<uint> gVoxCounters             : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_COUNTERS);
RWStructuredBuffer<int>  gVoxelClusterAssignments : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_CLUSTER_ASSIGN);
RWStructuredBuffer<int>  gClusterSeedCompactIds   : BAMBOO_PASS_UAV(DEBUG_VIEW_REG_CLUSTER_SEEDS);

cbuffer VoxelGridCB : BAMBOO_PASS_CBV(REG_VOXEL_GRID_CB)
{
    float3 voxGridMin;
    float  voxVoxelSize;
    float3 voxGridMax;
    uint   voxGridDim;
    // Order matches VoxelGridConstants exactly, and nothing checks that it does —
    // a swapped pair here is a silent wrong-value read, not a compile error.
    uint   voxInjectUseAvg;
    uint   _voxReserved0;
    float  voxHeatScale;
};

cbuffer DebugViewCB : BAMBOO_PASS_CBV(DEBUG_VIEW_REG_CB)
{
    uint gDebugView;
    uint gOutputWidth;
    uint gOutputHeight;
    uint gPad;
};

// Same ramp as the raster overlay so a view reads identically either way.
float3 HeatColor(float t)
{
    t = saturate(t);
    float3 c = lerp(float3(0, 0, 0), float3(1, 0, 0), saturate(t * 3.0));
    c = lerp(c, float3(1, 1, 0), saturate((t - 0.33) * 3.0));
    c = lerp(c, float3(1, 1, 1), saturate((t - 0.66) * 3.0));
    return c;
}

float3 HashColor(int3 v, uint3 primes)
{
    return float3(
        float((uint(v.x) * primes.x) & 0xFFu) / 255.0,
        float((uint(v.y) * primes.y) & 0xFFu) / 255.0,
        float((uint(v.z) * primes.z) & 0xFFu) / 255.0);
}

float3 PaintVoxelViews(float3 posW)
{
    int3 voxelCoord = int3(floor((posW - voxGridMin) / voxVoxelSize));
    if (any(voxelCoord < 0) || any(voxelCoord >= int(voxGridDim)))
        return float3(0.05, 0.05, 0.05);

    // The MRCS column clustering: which of the 32 clusters this voxel landed in.
    // Grouping is by 128-bit visibility fingerprint plus intensity, with the
    // position term weighted zero (vxpgCluster.hlsl), so the patches follow what
    // sees the same light rather than the grid — a view that looked like grid
    // blocks would be the bug.
    if (gDebugView == BUFFER_VIEW_VOXEL_CLUSTERS)
    {
        const uint flatId = uint(voxelCoord.x) + uint(voxelCoord.y) * voxGridDim
                          + uint(voxelCoord.z) * voxGridDim * voxGridDim;
        const int compactId = gVoxInverseIndex[flatId];
        if (compactId < 0)
            return float3(0.0, 0.0, 0.15); // unlit voxel: never entered the compacted set

        if (uint(compactId) >= gVoxCounters[0])
            return float3(1.0, 0.0, 1.0); // compact id past the live count = corruption

        [loop] for (uint seed = 0; seed < 32u; ++seed)
            if (gClusterSeedCompactIds[seed] == compactId)
                return float3(1.0, 1.0, 1.0);

        const int cluster = gVoxelClusterAssignments[compactId];
        if (cluster < 0 || cluster >= 32)
            return float3(1.0, 0.0, 1.0);

        // Same hue wheel as the guided technique's view 9, so the two agree:
        // golden-ratio step puts neighbouring cluster ids far apart in hue.
        const float hue = frac(float(cluster) * 0.618034);
        const float h6  = hue * 6.0;
        return saturate(float3(abs(h6 - 3.0) - 1.0, 2.0 - abs(h6 - 2.0), 2.0 - abs(h6 - 4.0)));
    }

    if (gDebugView == BUFFER_VIEW_VOXEL_IRRADIANCE)
    {
        uint vplCount = gVoxelVplCount[voxelCoord];
        if (vplCount == 0u)
            return float3(0.05, 0.05, 0.05);
        float irradiance = (float(gVoxelIrradiance[voxelCoord]) / 100.0) / float(vplCount);
        return HeatColor(1.0 - exp(-irradiance * voxHeatScale));
    }

    if (gVoxelOccupancy[voxelCoord] == 0u)
        return float3(0.05, 0.05, 0.05);
    return HashColor(voxelCoord, uint3(73u, 151u, 211u));
}

float3 PaintSuperpixelViews(uint2 pixel)
{
    uint width, height;
    gSuperpixelIndex.GetDimensions(width, height);
    int id = gSuperpixelIndex[int2(pixel)];
    if (id < 0)
        return float3(0, 0, 0);

    if (gDebugView == BUFFER_VIEW_SUPERPIXEL_ID)
        return float3(
            float((uint(id) * 131u + 17u) & 0xFFu) / 255.0,
            float((uint(id) * 197u + 71u) & 0xFFu) / 255.0,
            float((uint(id) * 53u + 113u) & 0xFFu) / 255.0);

    const int SUPERPIXEL_SIZE = 32;
    int mapX = (int(width) + SUPERPIXEL_SIZE - 1) / SUPERPIXEL_SIZE;
    int2 sp2D = int2(id % mapX, id / mapX);
    float3 n = Unorm32OctahedronToUnitVector(asuint(gSuperpixelCenter[sp2D].w));
    return n * 0.5 + 0.5;
}

[numthreads(8, 8, 1)]
void DebugViewPaint(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gOutputWidth || pixel.y >= gOutputHeight)
        return;

    float3 color = float3(0, 0, 0);

    if (gDebugView == BUFFER_VIEW_SUPERPIXEL_ID || gDebugView == BUFFER_VIEW_SUPERPIXEL_REP)
    {
        color = PaintSuperpixelViews(pixel);
    }
    else
    {
        // Every remaining view starts from the primary hit the shared VBuffer found.
        float4 shadingPoint = gShadingPoints[int2(pixel)];
        const bool missed = shadingPoint.x > 1e29;

        if (missed)
            color = float3(0, 0, 0);
        else if (gDebugView == BUFFER_VIEW_SHADING_NORMAL)
            color = Unorm32OctahedronToUnitVector(asuint(shadingPoint.w)) * 0.5 + 0.5;
        else if (gDebugView == BUFFER_VIEW_SHADING_POS)
            color = saturate((shadingPoint.xyz - voxGridMin) / max(voxGridMax - voxGridMin, 1e-4));
        else if (voxGridDim > 0)
            color = PaintVoxelViews(shadingPoint.xyz);
    }

    gDebugOutput[int2(pixel)] = float4(color, 1.0);
}
