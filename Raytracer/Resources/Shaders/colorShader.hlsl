#define MAX_TEXTURES 512
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
    float4x4 worldInvTranspose;
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

RWTexture3D<uint> gVoxelOccupancy  : register(u1);
RWTexture3D<uint> gVoxelIrradiance : register(u2);
RWTexture3D<uint> gVoxelVplCount   : register(u3);
RWTexture2D<float4> gShadingPoints : register(u4); // VXPG primary G-buffer (pos, octaN)

cbuffer VoxelGridCB : register(b4)
{
    float3 voxGridMin;
    float  voxVoxelSize;
    float3 voxGridMax;
    uint   voxGridDim;
    uint   voxInjectUseAvg;
    uint   _voxReserved0;
    float  voxHeatScale;
    uint   _voxPad0;
}

// Black -> red -> yellow -> white heat ramp, t in [0, 1]
float3 HeatColor(float t)
{
    t = saturate(t);
    float3 c;
    c.r = saturate(t * 3.0);
    c.g = saturate(t * 3.0 - 1.0);
    c.b = saturate(t * 3.0 - 2.0);
    return c;
}

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

    // Build TBN — re-orthogonalize T against N to fix interpolation drift
    float3 N = normalize(pin.NormalW);
    float3 T = normalize(pin.TangentW.xyz - dot(pin.TangentW.xyz, N) * N);
    float3 B = cross(N, T) * pin.TangentW.w;
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

    if (debugMode == 11 || debugMode == 12)
    {
        // ShadingPoints G-buffer (VXPG): primary worldPos + octahedral normal,
        // emitted by light injection. Sample via screen UV * texture dims so it
        // maps correctly even if the G-buffer res differs from the backbuffer.
        uint spW, spH;
        gShadingPoints.GetDimensions(spW, spH);
        int2 spTexel = int2(pin.ScreenUV * float2(spW, spH));
        float4 sp = gShadingPoints[spTexel];
        if (sp.x > 1e29) return float4(0, 0, 0, 1); // invalid (primary miss)
        if (debugMode == 11)
        {
            float3 n = Unorm32OctahedronToUnitVector(asuint(sp.w));
            return float4(n * 0.5 + 0.5, 1.0);
        }
        // Position remapped into the voxel grid bounds for a readable color
        float3 t = (sp.xyz - voxGridMin) / max(voxGridMax - voxGridMin, 1e-4);
        return float4(saturate(t), 1.0);
    }

    if ((debugMode == 8 || debugMode == 9 || debugMode == 10) && voxGridDim > 0)
    {
        int3 vidx = int3(floor((pin.PosW - voxGridMin) / voxVoxelSize));
        if (all(vidx >= 0) && all(vidx < int(voxGridDim)))
        {
            if (debugMode == 10)
            {
                // Supervoxels — hash-colored coarse cell (voxel / clusterFactor).
                // SUPERVOXEL_GRID_FACTOR must match Constants::Graphics.
                const int factor = 16;
                // Surface points near a voxel boundary can floor into an empty
                // neighbor (conservative-raster coverage gap, worse at high res).
                // Search the 3x3x3 neighborhood for the nearest occupied voxel.
                int3 found = int3(-1, -1, -1);
                [unroll] for (int dz = -1; dz <= 1; ++dz)
                [unroll] for (int dy = -1; dy <= 1; ++dy)
                [unroll] for (int dx = -1; dx <= 1; ++dx)
                {
                    int3 n = vidx + int3(dx, dy, dz);
                    if (all(n >= 0) && all(n < int(voxGridDim)) && gVoxelOccupancy[n] != 0u)
                        found = n;
                }
                if (found.x >= 0)
                {
                    // Supervoxel id -> high-contrast hash color.
                    int3 sv = found / factor;
                    float3 cellColor = float3(
                        float((uint(sv.x) * 131u + 17u) & 0xFFu) / 255.0,
                        float((uint(sv.y) * 197u + 71u) & 0xFFu) / 255.0,
                        float((uint(sv.z) * 53u + 113u) & 0xFFu) / 255.0);
                    // Per-voxel 3D checkerboard -> low-contrast brightness wobble
                    // so individual voxels stay readable within a supervoxel.
                    // Parity from the true surface cell (vidx), not the
                    // neighbor-filled cell, so adjacent voxels always alternate.
                    float checker = ((vidx.x + vidx.y + vidx.z) & 1) ? 1.0 : 0.85;
                    return float4(cellColor * checker, 1.0);
                }
            }
            else if (debugMode == 9)
            {
                // VoxelIrradiance — heat map of injection pass output
                uint packedIrr = gVoxelIrradiance[vidx];
                uint count = gVoxelVplCount[vidx];
                if (count > 0u)
                {
                    float irradiance = (float(packedIrr) / 100.0) / float(count);
                    float t = 1.0 - exp(-irradiance * voxHeatScale);
                    return float4(HeatColor(t), 1.0);
                }
            }
            else
            {
                // Voxels — hash-colored occupancy
                uint occ = gVoxelOccupancy[vidx];
                if (occ != 0u)
                {
                    float3 cellColor = float3(
                        float((vidx.x * 73u) & 0xFFu) / 255.0,
                        float((vidx.y * 151u) & 0xFFu) / 255.0,
                        float((vidx.z * 211u) & 0xFFu) / 255.0);
                    return float4(cellColor, 1.0);
                }
            }
        }
        return float4(0.05, 0.05, 0.05, 1.0);
    }

    float4 debugResult = ApplyRasterDebugMode(debugMode, debugData);
    if (debugResult.x >= 0) return debugResult;

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