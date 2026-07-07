// Shared between C++ and HLSL
#ifdef __cplusplus
#pragma once

#include "DebugViewDoc.h"
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
	VoxelOccupancy = 8,
	VoxelIrradiance = 9,
	Supervoxels = 10,
	ShadingPointsNormal = 11,
	ShadingPointsPos = 12,
	TangentHealth = 13,
	SuperpixelId = 15,
	SuperpixelRepresentative = 16,
};

// Runtime docs, one per enum entry in order (FormatDebugViewDocs static_asserts the count).
inline constexpr DebugViewDoc kRasterDebugModeDocs[] = {
	{"nothing", "normal shaded image", "debug path disabled"},
	{"material textures", "flat texture colors, no lighting", "outputs sampled base color"},
	{"normal mapping", "smooth RGB direction colors (XYZ mapped to RGB)", "outputs the normal-mapped shading normal"},
	{"vertex data / model import", "faceted RGB direction colors, no texture detail", "outputs the interpolated vertex normal"},
	{"normal map sampling", "purple-ish tangent-space texture detail", "outputs the raw normal map sample"},
	{"tangent import / mikktspace", "smooth RGB direction colors following UV seams", "outputs the vertex tangent"},
	{"UV unwrap", "red-green gradients per chart, no distortion", "outputs texcoords as RG"},
	{"roughness texture", "grayscale: dark = smooth, bright = rough", "outputs the roughness channel"},
	{"VoxelizationPass", "geometry-conforming voxel shell, checkerboard colors", "shades surfaces whose voxel is occupied"},
	{"LightInjectionPass irradiance", "heat ramp bright on lit surfaces, dark in shadow", "reads injected voxel irradiance at the surface voxel"},
	{"grid-cell supervoxels (analytic)", "coarse colored blocks over geometry", "hash-colors voxelCoord / supervoxelFactor at the surface voxel"},
	{"LightInjectionPass ShadingPoints G-buffer", "same direction colors as WorldNormals, black where primary ray missed", "decodes the octahedral normal from the ShadingPoints texture"},
	{"LightInjectionPass ShadingPoints G-buffer", "world-position gradient, black where primary ray missed", "visualizes ShadingPoints.xyz scaled into color"},
	{"tangent quality (NaN regression sentinel)", "all green; red = tangent parallel to normal, magenta = NaN", "flags degenerate tangent frames that produced the blue-artifact bug"},
	{"SuperpixelPass (SLIC)", "~32px mosaic cells hugging geometry edges", "hash-colors the per-pixel superpixel id"},
	{"SuperpixelPass (SLIC)", "mosaic of direction colors, one normal per cell", "paints each pixel with its superpixel's representative normal"},
};

// The furthest VXPG stage a raster debug view needs to read its data.
inline VxpgStage StageFor(RasterDebugMode mode)
{
	switch (mode)
	{
	case RasterDebugMode::VoxelOccupancy:   // reads occupancy
	case RasterDebugMode::Supervoxels:      // reads occupancy (supervoxel id is analytic)
		return VxpgStage::Voxelize;
	case RasterDebugMode::SuperpixelId:            // reads u_index
	case RasterDebugMode::SuperpixelRepresentative: // reads u_index + u_center
		return VxpgStage::Superpixel;
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
