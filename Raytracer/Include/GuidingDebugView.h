#pragma once

#include "DebugViewDoc.h"

// Debug visualization for the guided path tracing technique.
// Encoded into PassConstants::guidingFlags bits 1-4 (see guidedPathTracing.hlsl).
// NOTE: 4 bits (views 0-15). The next view past 15 needs a wider bitfield.
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
	FingerprintView = 8,
	ClusterView = 9,
	ClusterVisibilityView = 10,
	LightTreeView = 11,
	TopLevelHeapView = 12,
	PdfRoundTripView = 13,
	SelectedClusterView = 14,
};

// Runtime docs, one per enum entry in order (FormatDebugViewDocs static_asserts the count).
inline constexpr DebugViewDoc kGuidingDebugViewDocs[] = {
	{"nothing", "normal guided-PT image", "debug path disabled"},
	{"GuidedPathTracingPass MIS (bias isolation)", "near-final image slightly dimmed where the guide also claims coverage; NOT a complete estimator — this is the BSDF MIS PART (view1 + view2 - direct = final)", "first bounce traces only the BSDF strategy; the sample keeps its MIS weight"},
	{"GuidedPathTracingPass MIS (bias isolation)", "dim image = direct light + only the indirect energy arriving from lit clusters visible to each superpixel; hard black shadows where the guide is dead/gated = expected (BSDF part fills them in the full render); NOT a complete estimator — this is the guide MIS PART", "first bounce traces only the guide strategy; the sample keeps its MIS weight"},
	{"GuidedPathTracingPass MIS weights", "red where BSDF strategy dominates, green where guide dominates", "false-color: R = BSDF strategy MIS weight, G = guide strategy MIS weight"},
	{"GuidedPathTracingPass guide sampling", "mostly green; red = wasted rays (gate-rejected), blue = rejected before tracing", "classifies each guided sample: accepted / hit outside chosen voxel / below horizon or zero pdf"},
	{"VoxelGuidingBuildPass inverse index (Pass 1)", "green on surfaces in active voxels, black elsewhere, ZERO red", "round-trip check: voxelID -> gInverseIndex -> compactID -> gCompactIds must map back to the same voxelID"},
	{"LightInjectionPass representative VPL (Pass 2)", "green on surfaces in active voxels, black elsewhere, ZERO red/magenta (disable accumulation for a crisp read)", "reads gVoxelRepresentative at the primary hit's voxel: red = active voxel missing data, magenta = stored position outside the voxel, green = OK"},
	{"LightInjectionPass per-pixel VPL buffer (Pass 2)", "noisy direction colors (second-bounce world), black where the VPL bounce missed", "decodes gVplPosition's octahedral normal per pixel straight from raygen, no trace"},
	{"VxpgFingerprintPass voxel fingerprints (Pass 3)", "mid-gray spatially-coherent interior surfaces, dark blue on unlit/sky, ZERO magenta; 128 green dots scattering per frame; all-black or all-white = shadow-ray or facing bug", "grayscale = popcount(fingerprint)/128 at the primary hit's voxel; magenta = bad inverse-index; green = the sampled representative pixels"},
	{"VxpgClusterPass cluster assignments (Pass 4)", "DISABLE ACCUMULATION (colors reshuffle per frame). Per frame: saturated hue-wheel regions, coherent-ISH with voxel-level interleaving at boundaries (clusters live in fingerprint+intensity space, not position - speckle inherits the fingerprint checkerboard); 32 single-voxel white dots on lit geometry = the seeds; ZERO magenta; pure structureless noise = assignment bug, single color = distance/center bug, no white dots = seeding bug", "categorical color = gVoxelClusterAssignments at the primary hit's voxel; white = seed voxel; magenta = bad inverse-index or unassigned; dark blue = unlit/sky"},
	{"VxpgClusterVisibilityPass mask (Pass 5)", "blocky 32px-tile grayscale: bright in open areas, darker in occluded pockets (behind curtains); all-black = check kernel dead / no VPLs, all-white = mask OR-saturation bug", "grayscale = popcount(gClusterVisibilityMask)/32 at the pixel's superpixel tile = how many of the 32 light clusters the screen region can see"},
	{"VxpgLightTreePass bottom tree (Pass 6)", "DISABLE ACCUMULATION. green on lit geometry = leaf walks up to the root AND its cluster root sits on that ancestor path; cyan = voxel has no cluster (assignment -1, faithful k-means++ outcome when all 32 centers are too far - fine in moderation); yellow = root reached but cluster root NOT an ancestor (cluster-root bookkeeping bug); red = parent-walk failed to reach the root in 64 steps; magenta = bad compact->leaf mapping / vx_idx round-trip fail / uint16-overflow frame; dark blue = unlit/sky. Mostly-green = healthy; any red = pointer cycle/corruption", "walks gLightTreeNodes from the hit voxel's leaf (via gCompactToLeaf) to the root, checking gClusterRootNodes[cluster] is on the path"},
	{"VxpgLightTreePass top-level tree (Pass 7)", "DISABLE ACCUMULATION. Screen-space per-superpixel (32px blocky). green = heap sum invariant holds (parent == sum of children for slots 1..31) and root > 0; magenta = invariant broken (wave-reduction translation bug); red = NaN/inf in the heap; dark blue = root 0 (superpixel sees no lit cluster / sky). Mostly green+blue = healthy; any magenta = the top-level reduction is wrong", "verifies the per-superpixel implicit importance heap gSpixelClusterImportanceHeap: heap[i] must equal heap[2i]+heap[2i+1] for the 31 internal slots"},
	{"GuidedPathTracingPass forward/reverse pdf round trip (integrator)", "DISABLE ACCUMULATION. green on lit geometry = forward walk pdf matches the reverse telescoped query and the leaf/cluster maps round-trip; magenta = MISMATCH — a silent-bias bug the image alone would never show (also covers heap-picked cluster with no tree root / dead branch); red = NaN/inf in either pdf; dark blue = guide dead here (heap root 0 / no lit voxels / no superpixel). ANY magenta = fix before trusting benchmarks", "samples the discrete guide chain once (heap walk + tree walk), then reverse-queries the same outcome via inverse index / cluster assignments / compact->leaf; pdfs compared at rel-eps 1e-3"},
	{"GuidedPathTracingPass forward-selected cluster (integrator)", "DISABLE ACCUMULATION (colors strobe per frame). Superpixel-blocky hue-wheel regions whose boundaries line up with view 10's tiles; magenta = heap picked a cluster with no tree root; dark blue = guide dead here. Uniform single color everywhere = spixel-index wiring bug; pixel-fine noise = index texture not superpixel-granular", "one importance-heap walk per pixel per frame, painted in the view-9 hue wheel — validates the superpixel index binding and per-region adaptation"},
};
