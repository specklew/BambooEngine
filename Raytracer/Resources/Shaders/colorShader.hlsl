#define MAX_TEXTURES 512

SamplerState gsamPointWrap : register(s0);
SamplerState gsamPointClamp : register(s1);
SamplerState gsamLinearWrap : register(s2);
SamplerState gsamLinearClamp : register(s3);
SamplerState gsamAnisotropicWrap : register(s4);
SamplerState gsamAnisotropicClamp : register(s5);

float3 directionOfLight = normalize(float3(-0.5, -0.5, -0.5));

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
    int textureIndex;
    int normalTextureIndex;
    int roughnessTextureIndex;
}

cbuffer PassConstants : register(b3)
{
    float uvX;
    float uvY;
}

Texture2D gTextures[MAX_TEXTURES] : register(t3, space0);

ByteAddressBuffer g_indices : register(t2);

struct VertexIn
{
    float3 PosL  : POSITION;
    float3 NormalL : NORMAL;
    float3 TangentL : TANGENT;
    float2 TexCoord : TEXCOORD;
    //float4 Color : COLOR;
};

struct VertexOut
{
    float4 PosH  : SV_POSITION;
    float3 NormalW : NORMAL;
    float3 TangentW : TANGENT;
    float2 TexCoord : TEXCOORD;
};

VertexOut vertex(VertexIn vin)
{
    VertexOut vout;

    vout.NormalW = vin.NormalL;
    vout.TexCoord = vin.TexCoord;
    
    // Transform to world space.

    // TODO: Something must not be right here. The multiplication should be reversed? (curr: world * posL, should it be posL * world)
    float4 posL = float4(vin.PosL, 1.0f);
    vout.PosH = mul(world, posL);
    vout.PosH = mul(vout.PosH, viewProj);

    float4 normalL = float4(vin.NormalL, 1.0f);
    vout.NormalW = mul(world, normalL);

    float4 tangentL = float4(vin.TangentL, 1.0f);
    vout.TangentW = mul(world, tangentL);
    
    return vout;
}

float4 pixel(VertexOut pin) : SV_Target
{
    float4 textureAlbedo = float4(1,0,1,1);
    float4 textureNormal = float4(0,0,0,1);
    if (textureIndex != -1)
    {
        textureAlbedo = gTextures[textureIndex].Sample(gsamLinearWrap, pin.TexCoord + float2(uvX, uvY));
    }
    if (normalTextureIndex != -1)
    {
        textureNormal = gTextures[normalTextureIndex].Sample(gsamLinearWrap, pin.TexCoord + float2(uvX, uvY));
    }

    float3 dir = normalize(float3(1, 0.5, 0.5));
    float3 normal = pin.NormalW;
    float3 tangent = pin.TangentW;
    float3 bitangent = cross(normal, tangent);

    float4x4 TBN = float4x4(
        tangent.x, bitangent.x, normal.x, 0,
        tangent.y, bitangent.y, normal.y, 0,
        tangent.z, bitangent.z, normal.z, 0,
        0,         0,           0,       1
    );

    TBN = transpose(TBN);
    
    float4 newNormal = normalize(mul(textureNormal, TBN));
    
    return float4(tangent.xyz, 1);
}