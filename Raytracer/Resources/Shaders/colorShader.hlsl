#define MAX_TEXTURES 256

SamplerState gsamPointWrap : register(s0);
SamplerState gsamPointClamp : register(s1);
SamplerState gsamLinearWrap : register(s2);
SamplerState gsamLinearClamp : register(s3);
SamplerState gsamAnisotropicWrap : register(s4);
SamplerState gsamAnisotropicClamp : register(s5);

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

cbuffer Material : register(b2)
{
    float4 ambient;
    uint textureIndex;
}

Texture2D gAlbedoTextures[MAX_TEXTURES] : register(t1, space0);

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
    float4 textureAlbedo = float4(1,1,1,1);
    if (textureIndex != -1)
    {
        textureAlbedo = gAlbedoTextures[textureIndex].Sample(gsamLinearWrap, float2(0.9f, 0.1f));
    }
    return ambient * textureAlbedo/*pin.Color*/;
}