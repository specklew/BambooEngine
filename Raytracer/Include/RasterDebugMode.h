// Shared between C++ and HLSL
#ifdef __cplusplus
#pragma once

#include "VxpgStage.h"

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
	TangentHealth = 13,
};

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
	float tangentDotN;    // |dot(tangent, N)|; near 1 = tangent parallel to N
	float normalNaN;      // vertex normal normalize is NaN
	float worldNormalNaN; // normal-mapped shading normal is NaN
};

#endif
