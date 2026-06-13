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

// Surface inputs for the debug views (rendered by TryDebugView in DebugViews.hlsl).
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

#endif
