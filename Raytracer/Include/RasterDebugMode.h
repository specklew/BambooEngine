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
	Voxels = 8,
	VoxelIrradiance = 9,
	Supervoxels = 10,
	ShadingPointsNormal = 11,
	ShadingPointsPos = 12,
};

// How far down the linear VXPG pipeline a frame must run. Ordered: a higher
// stage implies every lower stage also runs (voxelize -> inject -> build ->
// cluster). Both debug views and raytracing techniques declare the furthest
// stage they need; the renderer runs the pipeline up to the maximum.
enum class VxpgStage : int
{
	None = 0,
	Voxelize,
	Inject,
	GuidingBuild,
	Supervoxel,
};

inline bool operator>=(VxpgStage a, VxpgStage b)
{
	return static_cast<int>(a) >= static_cast<int>(b);
}

// The furthest VXPG stage a raster debug view needs to read its data.
inline VxpgStage StageFor(RasterDebugMode mode)
{
	switch (mode)
	{
	case RasterDebugMode::Voxels:           // reads occupancy
	case RasterDebugMode::Supervoxels:      // reads occupancy (supervoxel id is analytic)
		return VxpgStage::Voxelize;
	case RasterDebugMode::VoxelIrradiance:  // reads injected irradiance
	case RasterDebugMode::ShadingPointsNormal:
	case RasterDebugMode::ShadingPointsPos: // read the injection G-buffer
		return VxpgStage::Inject;
	default:
		return VxpgStage::None;
	}
}

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
