// Shared between C++ and HLSL
#ifdef __cplusplus
#pragma once

#include "DebugViewDoc.h"

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
	TangentHealth = 13,
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
	{"tangent quality (NaN regression sentinel)",
	 "all green; red = tangent parallel to normal, magenta = NaN",
	 "flags degenerate tangent frames that produced the blue-artifact bug"},
	};

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
