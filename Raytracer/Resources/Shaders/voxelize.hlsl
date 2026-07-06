// Geometry bake (ADR 0004): conservative-raster the scene into the voxel grid
// ONCE per bake (scene load / grid resize / bound-flag change), not per frame.
// Three fixed-axis draws (x, y, z) with HW conservative raster; each fragment
// marks its center voxel occupied and injects a quantized AABB of the geometry
// inside that voxel (SIByL bake-injection.slang). At the shipped defaults
// (useCompact = clipping = 0) the injected AABB is the full voxel cube.
// SIByL's z_conservative flag is subsumed by the 3-axis draw scheme.

#include "voxelGrid.hlsli"
#include "TriangleClip.hlsl"

cbuffer VoxelGridCB : register(b0)
{
    VoxelGridParams gGrid;
}

cbuffer ModelTransforms : register(b1)
{
    float4x4 world;
    float4x4 worldInvTranspose;
}

cbuffer BakeCB : register(b2)
{
    uint axisIndex;
    uint useCompact; // 1 = tight AABB of the triangle sliver inside the voxel
    uint useClipping; // 1 = clip the triangle against the voxel before the AABB
    uint _pad0;
}

RWTexture3D<uint>        gOccupancy     : register(u0);
RWStructuredBuffer<uint> gBakedBoundMin : register(u1); // 4 uints per cell, quantized to the voxel cube
RWStructuredBuffer<uint> gBakedBoundMax : register(u2);

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

void pixel(VsOut pin, float3 bary : SV_Barycentrics)
{
    float3 posW0 = GetAttributeAtVertex(pin.PosW, 0);
    float3 posW1 = GetAttributeAtVertex(pin.PosW, 1);
    float3 posW2 = GetAttributeAtVertex(pin.PosW, 2);
    float3 posW  = posW0 * bary.x + posW1 * bary.y + posW2 * bary.z;

    int3 idx = WorldToVoxelIndex(gGrid, posW);
    if (!VoxelInBounds(gGrid, idx)) return;

    uint old;
    InterlockedOr(gOccupancy[idx], 1u, old);

    float3 triangleVoxelSpace[3];
    triangleVoxelSpace[0] = (posW0 - gGrid.gridMin) / gGrid.voxelSize;
    triangleVoxelSpace[1] = (posW1 - gGrid.gridMin) / gGrid.voxelSize;
    triangleVoxelSpace[2] = (posW2 - gGrid.gridMin) / gGrid.voxelSize;
    InjectTriangleVoxelBound(triangleVoxelSpace, idx);
}
