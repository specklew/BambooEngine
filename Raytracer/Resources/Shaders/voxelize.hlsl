// Geometry bake (ADR 0004): conservative-raster the scene into the voxel grid
// ONCE per bake (scene load / grid resize / bound-flag change), not per frame.
// Three fixed-axis draws (x, y, z) with HW conservative raster; each fragment
// marks its center voxel occupied and injects a quantized AABB of the geometry
// inside that voxel (SIByL bake-injection.slang). Bamboo defaults useCompact = 1
// (tight triangle bound) and clipping = 0; SIByL ships both off, which makes the
// injected AABB the full voxel cube.
// SIByL's z_conservative flag is subsumed by the 3-axis draw scheme.

#include "PassRegisters.h"
#include "voxelGrid.hlsli"
#include "TriangleClip.hlsl"

cbuffer VoxelGridCB : BAMBOO_PASS_CBV(VOXEL_BAKE_REG_GRID_CB)
{
    VoxelGridParams gGrid;
}

cbuffer ModelTransforms : BAMBOO_PASS_CBV(VOXEL_BAKE_REG_MODEL_CB)
{
    float4x4 world;
    float4x4 worldInvTranspose;
}

cbuffer BakeCB : BAMBOO_PASS_CBV(VOXEL_BAKE_REG_AXIS_CB)
{
    uint axisIndex;
    uint useCompact; // 1 = tight AABB of the triangle sliver inside the voxel
    uint useClipping; // 1 = clip the triangle against the voxel before the AABB
    uint _pad0;
}

RWTexture3D<uint>        gOccupancy     : BAMBOO_PASS_UAV(VOXEL_BAKE_REG_OCCUPANCY);
RWStructuredBuffer<uint> gBakedBoundMin : BAMBOO_PASS_UAV(VOXEL_BAKE_REG_BAKED_BOUND_MIN); // 4 uints per cell, quantized to the voxel cube
RWStructuredBuffer<uint> gBakedBoundMax : BAMBOO_PASS_UAV(VOXEL_BAKE_REG_BAKED_BOUND_MAX);

struct VsIn
{
    float3 PosL    : POSITION;
    float3 NormalL : NORMAL;
    float4 TangentL: TANGENT;
    float2 TexCoord: TEXCOORD;
};

struct VsOut
{
    float4 SvPos                : SV_Position;
    nointerpolation float3 PosW : POSITION0; // per-vertex via GetAttributeAtVertex
};

VsOut vertex(VsIn vin)
{
    VsOut vout;
    float4 posL = float4(vin.PosL, 1.0f);
    float3 posW = mul(posL, world).xyz;
    vout.PosW   = posW;
    vout.SvPos  = ProjectForAxis(gGrid, posW, axisIndex);
    return vout;
}

// Inject the triangle's bound into one voxel: optionally clip the triangle to
// the voxel cube, take the (clipped) AABB, quantize it to [0, 0xffffffff]
// relative to the voxel, and atomically merge (bake-injection.slang).
void InjectTriangleVoxelBound(float3 triangleVoxelSpace[3], int3 voxelId)
{
    float3 voxelMin = float3(voxelId);
    float3 voxelMax = voxelMin + 1.0f;

    float3 boundMin;
    float3 boundMax;
    if (useCompact != 0u)
    {
        int vertexCount = 3;
        float3 vertices[9];
        vertices[0] = triangleVoxelSpace[0];
        vertices[1] = triangleVoxelSpace[1];
        vertices[2] = triangleVoxelSpace[2];
        if (useClipping != 0u)
            ClipTriangleAgainstAABB(vertices, vertexCount, voxelMin, voxelMax);

        boundMin = float3(99999.0f, 99999.0f, 99999.0f);
        boundMax = -boundMin;
        for (int i = 0; i < vertexCount; ++i)
        {
            boundMin = min(boundMin, vertices[i]);
            boundMax = max(boundMax, vertices[i]);
        }
        boundMax = max(min(boundMax, voxelMax), voxelMin);
        boundMin = min(max(boundMin, voxelMin), voxelMax);
    }
    else
    {
        boundMin = voxelMin;
        boundMax = voxelMax;
    }

    const float maxUint = 4294967295.0f;
    uint3 quantizedMin = uint3(saturate(boundMin - voxelMin) * maxUint);
    uint3 quantizedMax = uint3(saturate(boundMax - voxelMin) * maxUint);

    uint flatId = uint(voxelId.x) + uint(voxelId.y) * gGrid.gridDim + uint(voxelId.z) * gGrid.gridDim * gGrid.gridDim;
    uint previous;
    InterlockedMin(gBakedBoundMin[flatId * 4 + 0], quantizedMin.x, previous);
    InterlockedMin(gBakedBoundMin[flatId * 4 + 1], quantizedMin.y, previous);
    InterlockedMin(gBakedBoundMin[flatId * 4 + 2], quantizedMin.z, previous);
    InterlockedMax(gBakedBoundMax[flatId * 4 + 0], quantizedMax.x, previous);
    InterlockedMax(gBakedBoundMax[flatId * 4 + 1], quantizedMax.y, previous);
    InterlockedMax(gBakedBoundMax[flatId * 4 + 2], quantizedMax.z, previous);
}

// Marks one voxel occupied and injects this triangle's bound into it.
void InjectIntoVoxel(float3 triangleVoxelSpace[3], int3 voxelId)
{
    if (!VoxelInBounds(gGrid, voxelId)) return;
    uint previous;
    InterlockedOr(gOccupancy[voxelId], 1u, previous);
    InjectTriangleVoxelBound(triangleVoxelSpace, voxelId);
}

void pixel(VsOut pin, float3 bary : SV_Barycentrics)
{
    float3 posW0 = GetAttributeAtVertex(pin.PosW, 0);
    float3 posW1 = GetAttributeAtVertex(pin.PosW, 1);
    float3 posW2 = GetAttributeAtVertex(pin.PosW, 2);
    float3 posW  = posW0 * bary.x + posW1 * bary.y + posW2 * bary.z;

    int3 idx = WorldToVoxelIndex(gGrid, posW);
    if (!VoxelInBounds(gGrid, idx)) return;

    float3 triangleVoxelSpace[3];
    triangleVoxelSpace[0] = (posW0 - gGrid.gridMin) / gGrid.voxelSize;
    triangleVoxelSpace[1] = (posW1 - gGrid.gridMin) / gGrid.voxelSize;
    triangleVoxelSpace[2] = (posW2 - gGrid.gridMin) / gGrid.voxelSize;

    // This fragment owns one voxel column: its cell in the two rasterized axes, the whole grid
    // along the axis being projected away. Clipping the triangle to that column gives the exact
    // range of voxels it crosses here.
    const uint depthAxis = axisIndex;
    float3 columnMin = float3(idx);
    float3 columnMax = columnMin + 1.0f;
    columnMin[depthAxis] = 0.0f;
    columnMax[depthAxis] = float(gGrid.gridDim);

    float3 clipped[9];
    clipped[0] = triangleVoxelSpace[0];
    clipped[1] = triangleVoxelSpace[1];
    clipped[2] = triangleVoxelSpace[2];
    int clippedCount = 3;
    ClipTriangleAgainstAABB(clipped, clippedCount, columnMin, columnMax);

    if (clippedCount < 1)
    {
        // The interpolated position can sit just outside the triangle (conservative
        // rasterization extrapolates attributes), leaving an empty clip. Keep the old behaviour
        // rather than dropping the fragment.
        InjectIntoVoxel(triangleVoxelSpace, idx);
        return;
    }

    float depthMin = clipped[0][depthAxis];
    float depthMax = depthMin;
    for (int i = 1; i < clippedCount; ++i)
    {
        depthMin = min(depthMin, clipped[i][depthAxis]);
        depthMax = max(depthMax, clipped[i][depthAxis]);
    }

    const int lastVoxel = int(gGrid.gridDim) - 1;
    const int firstDepth = clamp(int(floor(depthMin)), 0, lastVoxel);
    const int lastDepth  = clamp(int(floor(depthMax)), 0, lastVoxel);

    int3 voxelId = idx;
    for (int depth = firstDepth; depth <= lastDepth; ++depth)
    {
        voxelId[depthAxis] = depth;
        InjectIntoVoxel(triangleVoxelSpace, voxelId);
    }
}
