// Per-frame clear of the injection accumulators. Occupancy is NOT cleared —
// it is a bake output that persists until the next rebake (ADR 0004).

RWTexture3D<uint> gIrradiance : register(u0);
RWTexture3D<uint> gVplCount   : register(u1);

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

    gIrradiance[tid] = 0u;
    gVplCount[tid]   = 0u;
}
