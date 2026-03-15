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

    float4 normalL = float4(vin.NormalL, 0.0f);
    vout.NormalW = mul(world, normalL);

    float4 tangentL = float4(vin.TangentL, 0.0f);
    vout.TangentW = mul(world, tangentL);
    
    return vout;
}

float4 pixel(VertexOut pin) : SV_Target
{
    float4 textureAlbedo = float4(1, 0, 1, 1);
    float4 textureNormal = float4(0.5, 0.5, 1.0, 1.0); // flat normal fallback
    if (textureIndex != -1)
    {
        textureAlbedo = gTextures[textureIndex].Sample(gsamLinearWrap, pin.TexCoord + float2(uvX, uvY));
    }
    if (normalTextureIndex != -1)
    {
        textureNormal = gTextures[normalTextureIndex].Sample(gsamLinearWrap, pin.TexCoord + float2(uvX, uvY));
    }

    // Build TBN — re-orthogonalize T against N to fix interpolation drift
    float3 N = normalize(pin.NormalW);
    float3 T = normalize(pin.TangentW - dot(pin.TangentW, N) * N);
    float3 B = cross(N, T);
    float3x3 TBN = float3x3(T, B, N);

    // Decode tangent-space normal from [0,1] to [-1,1] and bring to world space
    float3 normalTS = textureNormal.xyz * 2.0 - 1.0;
    float3 worldNormal = normalize(mul(normalTS, TBN));

    float3 dir = normalize(float3(1, 0.5, 0.5));
    float diffuse = max(0.0, dot(dir, worldNormal));

    return float4(textureAlbedo.rgb * diffuse, 1.0);
}