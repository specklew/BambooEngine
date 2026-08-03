#include "PassRegisters.h"
#define MAX_TEXTURES FRAME_MAX_TEXTURES
#include "RasterDebugMode.h"
#include "passConstants.hlsl"
#include "consts.hlsl"
#include "Octahedral.hlsl"

SamplerState gsamPointWrap : register(s0);
SamplerState gsamPointClamp : register(s1);
SamplerState gsamLinearWrap : register(s2);
SamplerState gsamLinearClamp : register(s3);
SamplerState gsamAnisotropicWrap : register(s4);
SamplerState gsamAnisotropicClamp : register(s5);

cbuffer CameraParams : BAMBOO_PASS_CBV(RASTER_REG_CAMERA_CB)
{
    float4x4 viewProj;
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
}

cbuffer ModelTransforms : BAMBOO_PASS_CBV(RASTER_REG_MODEL_CB)
{
    float4x4 world;
    float4x4 worldInvTranspose;
}

cbuffer Material : BAMBOO_PASS_CBV(RASTER_REG_MATERIAL_CB)
{
    float4 baseColorFactor;
    int textureIndex;
    int normalTextureIndex;
    int roughnessTextureIndex;
    float metallicFactor;
    float roughnessFactor;
}

Texture2D gTextures[MAX_TEXTURES] : BAMBOO_PASS_SRV(RASTER_REG_TEXTURES);

ByteAddressBuffer g_indices : BAMBOO_PASS_SRV(RASTER_REG_INDICES);

#include "DebugViews.hlsl" // after DebugData; buffer views live in DebugViewPass now

struct VertexIn
{
    float3 PosL  : POSITION;
    float3 NormalL : NORMAL;
    float4 TangentL : TANGENT; // xyz = tangent, w = bitangent sign
    float2 TexCoord : TEXCOORD;
};

struct VertexOut
{
    float4 PosH  : SV_POSITION;
    float3 PosW  : POSITION;
    float3 NormalW : NORMAL;
    float4 TangentW : TANGENT; // xyz = tangent, w = bitangent sign
    float2 TexCoord : TEXCOORD;
    // Screen-space [0,1] UV (linear in screen space). Used to sample full-screen
    // resources (ShadingPoints G-buffer) independent of render-target resolution.
    noperspective float2 ScreenUV : TEXCOORD1;
};

VertexOut vertex(VertexIn vin)
{
    VertexOut vout;

    vout.TexCoord = vin.TexCoord;

    // Transform to world space (row-vector convention: v * M)
    float4 posL = float4(vin.PosL, 1.0f);
    float4 posW = mul(posL, world);
    vout.PosW = posW.xyz;
    vout.PosH = mul(posW, viewProj);
    // NDC.xy (clip.xy/clip.w) -> [0,1], flip Y for texture space. Linear in
    // screen space, so noperspective interpolation gives per-pixel screen UV.
    vout.ScreenUV = vout.PosH.xy / vout.PosH.w * float2(0.5, -0.5) + 0.5;

    // Normals require the inverse-transpose of the world matrix.
    // worldInvTranspose is uploaded as W^{-1} without CPU transpose, so HLSL
    // reads it as (W^{-1})^T. Using mul(M, v) applies (W^{-1})^T * n directly.
    float4 normalL = float4(vin.NormalL, 0.0f);
    vout.NormalW = mul(worldInvTranspose, normalL).xyz;

    // Tangent vectors transform with the world matrix (same as directions)
    float4 tangentL = float4(vin.TangentL.xyz, 0.0f);
    vout.TangentW = float4(mul(tangentL, world).xyz, vin.TangentL.w);

    return vout;
}

#include "BRDF.hlsl"

float4 pixel(VertexOut pin) : SV_Target
{
    float3 directionOfLight = normalize(float3(0.0f, -0.7071f, -0.7071f));
    
    float2 uv = pin.TexCoord + float2(uvX, uvY);

    // Sample textures
    float4 textureAlbedo = baseColorFactor;
    float4 textureNormal = float4(0.5, 0.5, 1.0, 1.0);
    float metallic = metallicFactor;
    float roughness = roughnessFactor;

    if (textureIndex != -1)
    {
        float4 texColor = gTextures[textureIndex].Sample(gsamLinearWrap, uv);
        texColor.rgb = pow(texColor.rgb, 2.2); // sRGB to linear
        textureAlbedo *= texColor;
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

    float3 N = normalize(pin.NormalW);

    // Mode-13 probes. Length is meaningless (world scale baked in), so test the NaN directly.
    float normalNaN = (N.x != N.x) ? 1.0 : 0.0;
    float rawTangentLength = length(pin.TangentW.xyz);
    float tangentDotN = rawTangentLength > 1e-12 ? abs(dot(pin.TangentW.xyz / rawTangentLength, N)) : 1.0;

    // Skip normal mapping when Gram-Schmidt collapses (matches SampleWorldSpaceNormal).
    float3 projectedTangent = pin.TangentW.xyz - dot(pin.TangentW.xyz, N) * N;
    float3 T = N;
    float3 worldNormal;
    if (dot(projectedTangent, projectedTangent) < 1e-8)
    {
        worldNormal = N;
    }
    else
    {
        T = normalize(projectedTangent);
        float3 B = cross(N, T) * pin.TangentW.w;
        float3x3 TBN = float3x3(T, B, N);
        float3 normalTS = textureNormal.xyz * 2.0 - 1.0;
        worldNormal = normalize(mul(normalTS, TBN));
    }

    DebugData debugData;
    debugData.albedo = textureAlbedo;
    debugData.worldNormal = worldNormal;
    debugData.vertexNormal = N;
    debugData.normalMap = textureNormal;
    debugData.tangent = T;
    debugData.uv = pin.TexCoord;
    debugData.roughness = roughness;
    debugData.metallic = metallic;
    debugData.tangentDotN = tangentDotN;
    debugData.normalNaN = normalNaN;
    debugData.worldNormalNaN = (worldNormal.x != worldNormal.x) ? 1.0 : 0.0;

    float4 debugResult;
    if (TryDebugView(debugMode, debugData, debugResult))
        return debugResult;

    // PBR shading
    float3 albedo = textureAlbedo.rgb;
    float3 L = -directionOfLight;
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

    float3 lightColor = float3(1.0, 1.0, 1.0);
    float3 Lo = (diffuse + specular) * lightColor * NdotL;

    // Ambient
    float3 ambient = float3(0.03, 0.03, 0.03) * albedo;
    float3 color = ambient + Lo;

    // ACES filmic tonemapping (Narkowicz 2015 fit)
    color = saturate((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14));

    // Linear to sRGB gamma
    color = pow(color, 1.0 / 2.2);

    return float4(color, textureAlbedo.a);
}