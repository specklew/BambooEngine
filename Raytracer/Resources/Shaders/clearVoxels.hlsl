RWTexture3D<uint> gOccupancy  : register(u0);
RWTexture3D<uint> gIrradiance : register(u1);
RWTexture3D<uint> gVplCount   : register(u2);

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

    gOccupancy[tid]  = 0u;
    gIrradiance[tid] = 0u;
    gVplCount[tid]   = 0u;
}
