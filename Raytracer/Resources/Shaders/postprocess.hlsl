Texture2D<float4>   gInput  : register(t0);
RWTexture2D<float4> gOutput : register(u0);

cbuffer PostProcessCB : register(b0)
{
    float exposure;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 id = tid.xy;
    float3 color = gInput[id].rgb * exposure;
    gOutput[id] = float4(saturate(color), 1.0);
}
