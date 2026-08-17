// Shared between C++ and HLSL
#ifdef __cplusplus
#pragma once

#include "DebugViewDoc.h"
#include "Utils/CVars.h"

// Debug views that only look at a buffer the VXPG chain already produced, so
// none of them needs anything from the integrator. That is what lets them be a
// graph node instead of a branch inside three different shaders: the node paints
// the image from the buffers, and the view works whichever technique is active
// (ADR 0017 phase 5b). World position comes from the ShadingPoints G-buffer, so
// visibility is the shared VBuffer's primary rays rather than a rasterized
// surface — same content, not the same pixels as the old raster-only versions.
enum class BufferDebugView : int
{
	None = 0,
	VoxelOccupancy = 1,
	VoxelIrradiance = 2,
	VoxelClusters = 3,
	ShadingPointsNormal = 4,
	ShadingPointsPos = 5,
	SuperpixelId = 6,
	SuperpixelRepresentative = 7,
};

// Runtime docs, one per enum entry in order (FormatDebugViewDocs static_asserts the count).
inline constexpr DebugViewDoc kBufferDebugViewDocs[] = {
	{"nothing", "the active technique's normal image", "debug path disabled"},
	{"VoxelizationPass occupancy", "geometry-conforming voxel shell, checkerboard colors",
	 "shades surfaces whose voxel is occupied"},
	{"LightInjectionPass irradiance", "heat ramp bright on lit surfaces, dark in shadow",
	 "reads injected voxel irradiance at the surface voxel"},
	{"VxpgClusterPass assignments", "up to 32 saturated hues in patches that follow shadowing rather than the grid; "
	 "white = a cluster seed voxel, magenta = unassigned or bad inverse index, dark blue = unlit voxel",
	 "voxel -> inverseIndex -> compactId -> clusterAssignments, hue-wheeled by cluster id"},
	{"LightInjectionPass ShadingPoints G-buffer", "smooth RGB direction colors, black where the primary ray missed",
	 "decodes the octahedral normal from the ShadingPoints texture"},
	{"LightInjectionPass ShadingPoints G-buffer", "world-position gradient, black where the primary ray missed",
	 "visualizes ShadingPoints.xyz scaled into the voxel grid's bounds"},
	{"SuperpixelPass (SLIC)", "~32px mosaic cells hugging geometry edges", "hash-colors the per-pixel superpixel id"},
	{"SuperpixelPass (SLIC)", "mosaic of direction colors, one normal per cell",
	 "paints each pixel with its superpixel's representative normal"},
};

// A buffer view runs the VXPG chain in place of the technique, so the renderer
// asks this rather than the technique whether the subgraph is needed.
inline AutoCVarEnum g_bufferDebugView("renderer.bufferDebugView", "Buffer debug visualization (any technique)",
                                      BufferDebugView::None, CVarFlags::None,
                                      FormatDebugViewDocs<BufferDebugView>(kBufferDebugViewDocs));

#else // HLSL

#define BUFFER_VIEW_VOXEL_OCCUPANCY   1
#define BUFFER_VIEW_VOXEL_IRRADIANCE  2
#define BUFFER_VIEW_VOXEL_CLUSTERS    3
#define BUFFER_VIEW_SHADING_NORMAL    4
#define BUFFER_VIEW_SHADING_POS       5
#define BUFFER_VIEW_SUPERPIXEL_ID     6
#define BUFFER_VIEW_SUPERPIXEL_REP    7

#endif
