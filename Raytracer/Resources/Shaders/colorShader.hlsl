cbuffer CameraParams : register(b0)
{
    float4x4 viewProj;
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
}

cbuffer ModelTransforms : register(b1)
{
    float4x4 world;
}

struct VertexIn
{
    float3 PosL  : POSITION;
    //float4 Color : COLOR;
};

struct VertexOut
{
    float4 PosH  : SV_POSITION;
    //float4 Color : COLOR;
};

VertexOut vertex(VertexIn vin)
{
    VertexOut vout;
    
    // Transform to world space.

    // TODO: Something must not be right here. The multiplication should be reversed? (curr: world * posL, should it be posL * world)
    float4 posL = float4(vin.PosL, 1.0f);
    vout.PosH = mul(world, posL);

    // Transform to homogeneous clip space.
    vout.PosH = mul(vout.PosH, viewProj);
    // Just pass vertex color into the pixel shader.
    //vout.Color = vin.Color;
    
    return vout;
}

float4 pixel(VertexOut pin) : SV_Target
{
    return float4(1,1,1,1)/*pin.Color*/;
}