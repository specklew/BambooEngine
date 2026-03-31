#define MAX_TEXTURES 512
#include "RasterDebugMode.h"
#include "passConstants.hlsl"

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
    float4 baseColorFactor;
    int textureIndex;
    int normalTextureIndex;
    int roughnessTextureIndex;
    float metallicFactor;
    float roughnessFactor;
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
    float3 PosW  : POSITION;
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
    float4 posW = mul(world, posL);
    vout.PosW = posW.xyz;
    vout.PosH = mul(posW, viewProj);

    float4 normalL = float4(vin.NormalL, 0.0f);
    vout.NormalW = mul(world, normalL);

    float4 tangentL = float4(vin.TangentL, 0.0f);
    vout.TangentW = mul(world, tangentL);
    
    return vout;
}

#include "BRDF.hlsl"

float4 pixel(VertexOut pin) : SV_Target
{
    float2 uv = pin.TexCoord + float2(uvX, uvY);

    // Sample textures
    float4 textureAlbedo = baseColorFactor;
    float4 textureNormal = float4(0.5, 0.5, 1.0, 1.0);
    float metallic = metallicFactor;
    float roughness = roughnessFactor;

    if (textureIndex != -1)
    {
        textureAlbedo *= gTextures[textureIndex].Sample(gsamLinearWrap, uv);
    }
    if (normalTextureIndex != -1)
    {
        textureNormal = gTextures[normalTextureIndex].Sample(gsamLinearWrap, uv);
    }
    if (roughnessTextureIndex != -1)
    {
        // glTF: green = roughness, blue = metallic
        float4 mr = gTextures[roughnessTextureIndex].Sample(gsamLinearWrap, uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }

    // Build TBN — re-orthogonalize T against N to fix interpolation drift
    float3 N = normalize(pin.NormalW);
    float3 T = normalize(pin.TangentW - dot(pin.TangentW, N) * N);
    float3 B = cross(N, T);
    float3x3 TBN = float3x3(T, B, N);

    // Decode tangent-space normal from [0,1] to [-1,1] and bring to world space
    float3 normalTS = textureNormal.xyz * 2.0 - 1.0;
    float3 worldNormal = normalize(mul(normalTS, TBN));

    // Debug visualization
    DebugData debugData;
    debugData.albedo = textureAlbedo;
    debugData.worldNormal = worldNormal;
    debugData.vertexNormal = N;
    debugData.normalMap = textureNormal;
    debugData.tangent = T;
    debugData.uv = pin.TexCoord;
    debugData.roughness = roughness;
    debugData.metallic = metallic;

    float4 debugResult = ApplyRasterDebugMode(debugMode, debugData);
    if (debugResult.x >= 0) return debugResult;

    // PBR shading
    float3 albedo = textureAlbedo.rgb;
    float3 L = normalize(float3(1, 0.5, 0.5));
    float3 V = normalize(cameraWorldPos - pin.PosW);
    float3 H = normalize(V + L);

    float NdotL = max(dot(worldNormal, L), 0.0);
    float NdotV = max(dot(worldNormal, V), 0.001);
    float NdotH = max(dot(worldNormal, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Dielectric F0 = 0.04, metals use albedo as F0
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float D = DistributionGGX(NdotH, roughness);
    float G = GeometrySmith(NdotV, NdotL, roughness);
    float3 F = FresnelSchlick(HdotV, F0);

    float3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 diffuse = kD * albedo / PI;

    float3 lightColor = float3(1.0, 1.0, 1.0) * 3.0;
    float3 Lo = (diffuse + specular) * lightColor * NdotL;

    // Ambient
    float3 ambient = float3(0.03, 0.03, 0.03) * albedo;
    float3 color = ambient + Lo;

    return float4(color, textureAlbedo.a);
}