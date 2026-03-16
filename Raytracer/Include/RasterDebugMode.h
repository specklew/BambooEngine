// Shared between C++ and HLSL
#ifdef __cplusplus
#pragma once

enum class RasterDebugMode : int
{
	None = 0,
	Albedo = 1,
	WorldNormals = 2,
	VertexNormals = 3,
	NormalMap = 4,
	Tangents = 5,
	UVs = 6,
	Roughness = 7,
};

#else // HLSL

struct DebugData
{
	float4 albedo;
	float3 worldNormal;
	float3 vertexNormal;
	float4 normalMap;
	float3 tangent;
	float2 uv;
	float roughness;
	float metallic;
};

float4 ApplyRasterDebugMode(int mode, DebugData d)
{
	if (mode == 1) return d.albedo;
	if (mode == 2) return float4(d.worldNormal * 0.5 + 0.5, 1);
	if (mode == 3) return float4(d.vertexNormal * 0.5 + 0.5, 1);
	if (mode == 4) return d.normalMap;
	if (mode == 5) return float4(d.tangent * 0.5 + 0.5, 1);
	if (mode == 6) return float4(d.uv, 0, 1);
	if (mode == 7) return float4(d.roughness, d.metallic, 0, 1);
	return float4(-1, -1, -1, -1);
}

#endif
