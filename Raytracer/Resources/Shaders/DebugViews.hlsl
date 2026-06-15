#ifndef DEBUG_VIEWS_HLSL
#define DEBUG_VIEWS_HLSL

// Include after the voxel resources, VoxelGridCB, HeatColor and DebugData are declared.

float4 DebugView_Surface(int mode, DebugData d)
{
    if (mode == 1) return d.albedo;
    if (mode == 2) return float4(d.worldNormal * 0.5 + 0.5, 1);
    if (mode == 3) return float4(d.vertexNormal * 0.5 + 0.5, 1);
    if (mode == 4) return d.normalMap;
    if (mode == 5) return float4(d.tangent * 0.5 + 0.5, 1);
    if (mode == 6) return float4(d.uv, 0, 1);
    if (mode == 7) return float4(d.roughness, d.metallic, 0, 1);
    if (mode == 13)
    {
        // Surface NaN sources. Red = shading normal, green = tangent parallel to N, blue = vertex normal.
        float parallel = saturate((d.tangentDotN - 0.9) * 10.0);
        return float4(d.worldNormalNaN, parallel, d.normalNaN, 1);
    }
    return float4(-1, -1, -1, -1);
}

float4 DebugView_ShadingPoints(int mode, float2 screenUV)
{
    uint gbufferWidth, gbufferHeight;
    gShadingPoints.GetDimensions(gbufferWidth, gbufferHeight);
    float4 shadingPoint = gShadingPoints[int2(screenUV * float2(gbufferWidth, gbufferHeight))];
    if (shadingPoint.x > 1e29) return float4(0, 0, 0, 1);
    if (mode == 11)
        return float4(Unorm32OctahedronToUnitVector(asuint(shadingPoint.w)) * 0.5 + 0.5, 1.0);
    float3 normalizedPos = (shadingPoint.xyz - voxGridMin) / max(voxGridMax - voxGridMin, 1e-4);
    return float4(saturate(normalizedPos), 1.0);
}

float4 DebugView_Voxels(int mode, float3 posW)
{
    int3 voxelCoord = int3(floor((posW - voxGridMin) / voxVoxelSize));
    if (all(voxelCoord >= 0) && all(voxelCoord < int(voxGridDim)))
    {
        if (mode == 10)
        {
            const int voxelsPerSupervoxel = 16;
            // Search 3x3x3 for nearest occupied voxel: surface points near a
            // boundary can floor into an empty neighbor (raster coverage gap).
            int3 occupiedCoord = int3(-1, -1, -1);
            [unroll] for (int dz = -1; dz <= 1; ++dz)
            [unroll] for (int dy = -1; dy <= 1; ++dy)
            [unroll] for (int dx = -1; dx <= 1; ++dx)
            {
                int3 neighborCoord = voxelCoord + int3(dx, dy, dz);
                if (all(neighborCoord >= 0) && all(neighborCoord < int(voxGridDim)) && gVoxelOccupancy[neighborCoord] != 0u)
                    occupiedCoord = neighborCoord;
            }
            if (occupiedCoord.x >= 0)
            {
                int3 supervoxelCoord = occupiedCoord / voxelsPerSupervoxel;
                float3 supervoxelColor = float3(
                    float((uint(supervoxelCoord.x) * 131u + 17u) & 0xFFu) / 255.0,
                    float((uint(supervoxelCoord.y) * 197u + 71u) & 0xFFu) / 255.0,
                    float((uint(supervoxelCoord.z) * 53u + 113u) & 0xFFu) / 255.0);
                // Parity from the true surface cell so adjacent voxels always alternate.
                float checker = ((voxelCoord.x + voxelCoord.y + voxelCoord.z) & 1) ? 1.0 : 0.85;
                return float4(supervoxelColor * checker, 1.0);
            }
        }
        else if (mode == 14)
        {
            // Data-driven supervoxel view: heat-ramp the mean irradiance stored by
            // the cluster pass (gSvIrradiance/gSvCount). Verifies the atomics that
            // mode 10 (analytic id hash) cannot. Same 3x3x3 occupied search so a
            // surface point maps to its occupied cell.
            int3 occupiedCoord = int3(-1, -1, -1);
            [unroll] for (int dz = -1; dz <= 1; ++dz)
            [unroll] for (int dy = -1; dy <= 1; ++dy)
            [unroll] for (int dx = -1; dx <= 1; ++dx)
            {
                int3 neighborCoord = voxelCoord + int3(dx, dy, dz);
                if (all(neighborCoord >= 0) && all(neighborCoord < int(voxGridDim)) && gVoxelOccupancy[neighborCoord] != 0u)
                    occupiedCoord = neighborCoord;
            }
            if (occupiedCoord.x >= 0)
            {
                uint factor   = max(voxSupervoxelFactor, 1u);
                uint svDim    = (voxGridDim + factor - 1u) / factor;
                uint3 svCoord = uint3(occupiedCoord) / factor;
                uint svIndex  = svCoord.x + svCoord.y * svDim + svCoord.z * svDim * svDim;
                if (svIndex >= MAX_SUPERVOXELS)
                    return float4(1.0, 0.0, 1.0, 1.0); // magenta = index overflow (cluster drops these)

                uint count = gSvCount[svIndex];
                if (count == 0u)
                    return float4(0.02, 0.02, 0.02, 1.0); // active voxel but no supervoxel sum = bug signal
                float meanIrradiance = (float(gSvIrradiance[svIndex]) / 100.0) / float(count);
                return float4(HeatColor(1.0 - exp(-meanIrradiance * voxHeatScale)), 1.0);
            }
        }
        else if (mode == 9)
        {
            uint vplCount = gVoxelVplCount[voxelCoord];
            if (vplCount > 0u)
            {
                float irradiance = (float(gVoxelIrradiance[voxelCoord]) / 100.0) / float(vplCount);
                return float4(HeatColor(1.0 - exp(-irradiance * voxHeatScale)), 1.0);
            }
        }
        else if (gVoxelOccupancy[voxelCoord] != 0u)
        {
            return float4(
                float((voxelCoord.x * 73u) & 0xFFu) / 255.0,
                float((voxelCoord.y * 151u) & 0xFFu) / 255.0,
                float((voxelCoord.z * 211u) & 0xFFu) / 255.0, 1.0);
        }
    }
    return float4(0.05, 0.05, 0.05, 1.0);
}

bool TryDebugView(int mode, DebugData d, float3 posW, float2 screenUV, out float4 color)
{
    color = float4(0, 0, 0, 1);

    if (mode == 11 || mode == 12)
    {
        color = DebugView_ShadingPoints(mode, screenUV);
        return true;
    }
    if ((mode == 8 || mode == 9 || mode == 10 || mode == 14) && voxGridDim > 0)
    {
        color = DebugView_Voxels(mode, posW);
        return true;
    }
    float4 surface = DebugView_Surface(mode, d);
    if (surface.x >= 0)
    {
        color = surface;
        return true;
    }
    return false;
}

#endif
