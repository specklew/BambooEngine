#pragma once

#include "DebugViewDoc.h"

// Debug visualization for the guided path tracing technique.
// Encoded into PassConstants::guidingFlags bits 1-3 (see guidedPathTracing.hlsl).
// NOTE: 3 bits are full (views 0-7). The next view needs a wider bitfield.
enum class GuidingDebugView : int
{
	None = 0,
	BsdfStrategyOnly = 1,
	GuideStrategyOnly = 2,
	MisWeights = 3,
	GuideAcceptance = 4,
	InverseIndexRoundTrip = 5,
	RepresentativeCheck = 6,
	VplPositionView = 7,
};

// Runtime docs, one per enum entry in order (FormatDebugViewDocs static_asserts the count).
inline constexpr DebugViewDoc kGuidingDebugViewDocs[] = {
	{"nothing", "normal guided-PT image", "debug path disabled"},
	{"GuidedPathTracingPass MIS (bias isolation)", "noisier image that converges to the same result as full MIS", "first bounce uses only the BSDF strategy, guide strategy disabled"},
	{"GuidedPathTracingPass MIS (bias isolation)", "image lit mostly where guiding targets bright voxels; converges to the same result", "first bounce uses only the guide strategy, BSDF strategy disabled"},
	{"GuidedPathTracingPass MIS weights", "red where BSDF strategy dominates, green where guide dominates", "false-color: R = BSDF strategy MIS weight, G = guide strategy MIS weight"},
	{"GuidedPathTracingPass guide sampling", "mostly green; red = wasted rays (gate-rejected), blue = rejected before tracing", "classifies each guided sample: accepted / hit outside chosen voxel / below horizon or zero pdf"},
	{"VoxelGuidingBuildPass inverse index (Pass 1)", "green on surfaces in active voxels, black elsewhere, ZERO red", "round-trip check: voxelID -> gInverseIndex -> compactID -> gCompactIds must map back to the same voxelID"},
	{"LightInjectionPass representative VPL (Pass 2)", "green on surfaces in active voxels, black elsewhere, ZERO red/magenta (disable accumulation for a crisp read)", "reads gVoxelRepresentative at the primary hit's voxel: red = active voxel missing data, magenta = stored position outside the voxel, green = OK"},
	{"LightInjectionPass per-pixel VPL buffer (Pass 2)", "noisy direction colors (second-bounce world), black where the VPL bounce missed", "decodes gVplPosition's octahedral normal per pixel straight from raygen, no trace"},
};
