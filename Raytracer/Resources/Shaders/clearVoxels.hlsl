RWTexture3D<uint>   gOccupancy : register(u0);
RWTexture3D<float4> gSh0       : register(u1);
RWTexture3D<float4> gSh1       : register(u2);
RWTexture3D<float4> gSh2       : register(u3);
RWTexture3D<float4> gSh3       : register(u4);
RWTexture3D<float4> gSh4       : register(u5);
RWTexture3D<float4> gSh5       : register(u6);
RWTexture3D<float4> gSh6       : register(u7);

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
    float4 zero = float4(0, 0, 0, 0);
    gSh0[tid] = zero;
    gSh1[tid] = zero;
    gSh2[tid] = zero;
    gSh3[tid] = zero;
    gSh4[tid] = zero;
    gSh5[tid] = zero;
    gSh6[tid] = zero;
}
