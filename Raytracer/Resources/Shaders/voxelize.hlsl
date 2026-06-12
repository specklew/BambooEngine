#include "voxelGrid.hlsli"

cbuffer VoxelGridCB : register(b0)
{
    VoxelGridParams gGrid;
}

cbuffer ModelTransforms : register(b1)
{
    float4x4 world;
    float4x4 worldInvTranspose;
}

cbuffer AxisCB : register(b2)
{
    uint axisIndex;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

RWTexture3D<uint> gOccupancy : register(u0);

struct VsIn
{
    float3 PosL    : POSITION;
    float3 NormalL : NORMAL;
    float4 TangentL: TANGENT;
    float2 TexCoord: TEXCOORD;
};

struct VsOut
{
    float4 SvPos  : SV_Position;
    float3 PosW   : POSITION0;
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

void pixel(VsOut pin)
{
    int3 idx = WorldToVoxelIndex(gGrid, pin.PosW);
    if (VoxelInBounds(gGrid, idx))
    {
        uint old;
        InterlockedOr(gOccupancy[idx], 1u, old);
    }
}
