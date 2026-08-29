// Pass-scoped shader registers: the counterpart to FrameBindingRegisters.h, which owns
// the frame layout. Included by C++ and by HLSL so a register number exists once
// (ADR 0019). Sections mirror passes; a pass's shader and its root signature read the
// same defines, so they cannot drift.
//
// Registers here are local to one root signature. Two passes reusing the same number for
// different resources is normal and not a collision — what matters is that everything
// bound by one signature agrees. Where several shaders share a signature (the guided
// integrator and its adaptive-q update; the two fingerprint kernels) that sharing is
// noted, because those numbers are not independent.
//
// Everything below lives in **space1**, declared with BAMBOO_PASS_CBV/_SRV/_UAV in HLSL
// and PassCbv/PassSrv/PassUav/PassTableEntry in C++ (ADR 0017 phase 4). space0 belongs to
// the frame layout alone, so these numbers are chosen freely: a pass register can never
// collide with a frame one, and adding a frame binding no longer means scanning every
// pass. The one exception is PassConstants (b3, space0), which is a frame binding that
// passes without the full frame layout still declare — passConstants.hlsl owns it.
#ifdef __cplusplus
#pragma once
#endif

#include "FrameBindingRegisters.h"

// Voxel grid constants, shared by every pass that reads the grid: the guided
// integrator, light injection and the rasterization debug views. Same number in
// all three signatures, so it is defined once here rather than per section.
#define REG_VOXEL_GRID_CB 0 // b

// ---------------------------------------------------------------------------
// DebugViewPass — debugViewPaint.hlsl. Reads VXPG products off the global heap
// and paints one full-screen image; no technique state, so it runs under any.
// ---------------------------------------------------------------------------
#define DEBUG_VIEW_REG_CB                 1 // b (b0 is REG_VOXEL_GRID_CB)
#define DEBUG_VIEW_REG_OUTPUT             0 // u
#define DEBUG_VIEW_REG_SHADING_POINTS     1 // u
#define DEBUG_VIEW_REG_VOXEL_OCCUPANCY    2 // u
#define DEBUG_VIEW_REG_VOXEL_IRRADIANCE   3 // u
#define DEBUG_VIEW_REG_VOXEL_VPL_COUNT    4 // u
#define DEBUG_VIEW_REG_SUPERPIXEL_INDEX   5 // u
#define DEBUG_VIEW_REG_SUPERPIXEL_CENTER  6 // u
// Root UAVs: the cluster products are structured buffers, not global-heap views.
#define DEBUG_VIEW_REG_INVERSE_INDEX      7 // u (root)
#define DEBUG_VIEW_REG_COUNTERS           8 // u (root)
#define DEBUG_VIEW_REG_CLUSTER_ASSIGN     9 // u (root)
#define DEBUG_VIEW_REG_CLUSTER_SEEDS     10 // u (root)

// ---------------------------------------------------------------------------
// FrameAccumulationPass — accumulation.hlsl
// ---------------------------------------------------------------------------
#define ACCUM_REG_CURRENT 0 // t
#define ACCUM_REG_ACCUM   0 // u
#define ACCUM_REG_DISPLAY 1 // u
#define ACCUM_REG_CB      0 // b
// Welford's M2 (sum of squared deviations) per pixel, so the estimator's own
// variance is measurable WITHOUT a reference image — the only way to separate
// "still noisy" from "converged to the wrong answer" on a technique with a known
// error floor. Behind renderer.accumulation.variance. A structured buffer rather
// than a texture so it can be a root UAV (root descriptors are buffers only) and
// so the reduction needs no descriptor table: xyz = M2 per channel, w = the
// running mean's luminance, which is what makes a relative variance possible
// without also binding the accumulation texture.
#define ACCUM_REG_VARIANCE_M2 2 // u (root) — u0/u1 are the table's accum + display
// Reduction of that buffer to one scalar pair, run at capture time only — never
// per frame, because reading it back means waiting for the GPU.
#define VARIANCE_REDUCE_REG_M2      0 // u (root)
#define VARIANCE_REDUCE_REG_RESULT  1 // u (root)
#define VARIANCE_REDUCE_REG_CB      0 // b

// ---------------------------------------------------------------------------
// PostProcessPass — postprocess.hlsl
// ---------------------------------------------------------------------------
#define POST_REG_INPUT  0 // t
#define POST_REG_OUTPUT 0 // u
#define POST_REG_CB     0 // b

// ---------------------------------------------------------------------------
// VoxelizationPass — voxelize.hlsl (bake), clearVoxels.hlsl, bakeClear.hlsl
// ---------------------------------------------------------------------------
#define VOXEL_BAKE_REG_GRID_CB         0 // b
#define VOXEL_BAKE_REG_MODEL_CB        1 // b
#define VOXEL_BAKE_REG_AXIS_CB         2 // b
#define VOXEL_BAKE_REG_OCCUPANCY       0 // u
#define VOXEL_BAKE_REG_BAKED_BOUND_MIN 1 // u
#define VOXEL_BAKE_REG_BAKED_BOUND_MAX 2 // u

#define VOXEL_FRAME_CLEAR_REG_CB          0 // b
#define VOXEL_FRAME_CLEAR_REG_IRRADIANCE  0 // u
#define VOXEL_FRAME_CLEAR_REG_VPL_COUNT   1 // u

#define VOXEL_BAKE_CLEAR_REG_CB              0 // b
#define VOXEL_BAKE_CLEAR_REG_OCCUPANCY       0 // u
#define VOXEL_BAKE_CLEAR_REG_BAKED_BOUND_MIN 1 // u
#define VOXEL_BAKE_CLEAR_REG_BAKED_BOUND_MAX 2 // u

// ---------------------------------------------------------------------------
// VoxelGuidingBuildPass — voxelGuidingBuild.hlsl
// ---------------------------------------------------------------------------
#define GUIDING_BUILD_REG_CB                    0  // b
#define GUIDING_BUILD_REG_IRRADIANCE            0  // u  (private heap)
#define GUIDING_BUILD_REG_VPL_COUNT             1  // u  (private heap)
#define GUIDING_BUILD_REG_VOXEL_REPRESENTATIVE  2  // u  (private heap)
#define GUIDING_BUILD_REG_COUNTERS              3  // u
#define GUIDING_BUILD_REG_COMPACT_IDS           4  // u
#define GUIDING_BUILD_REG_INVERSE_INDEX         5  // u
#define GUIDING_BUILD_REG_LIVE_BOUND_MIN        6  // u
#define GUIDING_BUILD_REG_LIVE_BOUND_MAX        7  // u
#define GUIDING_BUILD_REG_COMPACT_LIGHT_POINTS  8  // u
#define GUIDING_BUILD_REG_PREMUL_IRRADIANCE     9  // u
#define GUIDING_BUILD_REG_BAKED_BOUND_MIN       10 // u
#define GUIDING_BUILD_REG_BAKED_BOUND_MAX       11 // u

// ---------------------------------------------------------------------------
// VxpgFingerprintPass — vxpgFingerprint.hlsl. Two kernels, two root signatures;
// the visibility kernel rebinds u1-u3 to different resources on purpose.
// ---------------------------------------------------------------------------
#define FINGERPRINT_PRESAMPLE_REG_CB                  0 // b
#define FINGERPRINT_PRESAMPLE_REG_SHADING_POINTS      0 // u (frame G-buffer, private heap)
#define FINGERPRINT_PRESAMPLE_REG_REPRESENTATIVES     1 // u
#define FINGERPRINT_PRESAMPLE_REG_DISPATCH_ARGS       2 // u
#define FINGERPRINT_PRESAMPLE_REG_COUNTERS            3 // u

#define FINGERPRINT_VISIBILITY_REG_TLAS               0 // t
#define FINGERPRINT_VISIBILITY_REG_REPRESENTATIVES    1 // u
#define FINGERPRINT_VISIBILITY_REG_LIGHT_POINTS       2 // u
#define FINGERPRINT_VISIBILITY_REG_DISPATCH_ARGS      3 // u
#define FINGERPRINT_VISIBILITY_REG_FINGERPRINTS       4 // u
#define FINGERPRINT_VISIBILITY_REG_CB                 0 // b (probe mode, diagnostic)

// ---------------------------------------------------------------------------
// VxpgClusterPass — vxpgCluster.hlsl
// ---------------------------------------------------------------------------
#define CLUSTER_REG_CB                  0 // b
#define CLUSTER_REG_SEED_COMPACT_IDS    0 // u
#define CLUSTER_REG_CENTERS             1 // u
#define CLUSTER_REG_DISPATCH_ARGS       2 // u
#define CLUSTER_REG_FINGERPRINTS        3 // u
#define CLUSTER_REG_COMPACT_IDS         4 // u
#define CLUSTER_REG_PREMUL_IRRADIANCE   5 // u
#define CLUSTER_REG_ASSIGNMENTS         6 // u
// Diagnostic only: per-cluster population and the two distance terms, written
// while vxpg.cluster.dumpStats is armed and idle otherwise.
#define CLUSTER_REG_STATS               7 // u

// ---------------------------------------------------------------------------
// VxpgClusterVisibilityPass — vxpgClusterVisibility.hlsl
// ---------------------------------------------------------------------------
#define CVIS_REG_GRID_CB                 1  // b
#define CVIS_REG_CB                      2  // b
#define CVIS_REG_VPL_POSITION            1  // u
#define CVIS_REG_VBUFFER                 2  // u
#define CVIS_REG_SUPERPIXEL_INDEX        3  // u
#define CVIS_REG_SPIXEL_GATHERED         4  // u
#define CVIS_REG_SPIXEL_COUNTER          5  // u
#define CVIS_REG_VISIBILITY_MASK         6  // u
#define CVIS_REG_INVERSE_INDEX           7  // u
#define CVIS_REG_CLUSTER_ASSIGNMENTS     8  // u
#define CVIS_REG_GATHERED_LIGHT_POINTS   9  // u
#define CVIS_REG_LIGHT_POINT_COUNTS      10 // u
#define CVIS_REG_AVG_VISIBILITY          11 // u

// ---------------------------------------------------------------------------
// VxpgLightTreePass — vxpgLightTree.hlsl
// ---------------------------------------------------------------------------
#define LIGHT_TREE_REG_GRID_CB              0  // b
#define LIGHT_TREE_REG_TOP_LEVEL_CB         1  // b
#define LIGHT_TREE_REG_SORT_KEYS            0  // u
#define LIGHT_TREE_REG_NODES                1  // u
#define LIGHT_TREE_REG_LEAF_RANGES          2  // u
#define LIGHT_TREE_REG_COMPACT_TO_LEAF      3  // u
#define LIGHT_TREE_REG_CLUSTER_ROOTS        4  // u
#define LIGHT_TREE_REG_DISPATCH_ARGS        5  // u
#define LIGHT_TREE_REG_COMPACT_IDS          6  // u
#define LIGHT_TREE_REG_CLUSTER_ASSIGNMENTS  7  // u
#define LIGHT_TREE_REG_PREMUL_IRRADIANCE    8  // u
#define LIGHT_TREE_REG_COUNTERS             9  // u
#define LIGHT_TREE_REG_NODE_VISITED         10 // u
#define LIGHT_TREE_REG_AVG_VISIBILITY       11 // u
#define LIGHT_TREE_REG_IMPORTANCE_HEAP      12 // u
#define LIGHT_TREE_REG_VISIBILITY_MASK      13 // u
#define LIGHT_TREE_REG_INDIRECT_ARGS        14 // u
// Slots inside gIndirectDispatchArgs. This buffer exists SEPARATELY from
// gDispatchArgs because the tree kernels read gDispatchArgs as a UAV (for
// numValidVoxels) in the very dispatches whose group counts come from here, and
// one resource cannot be INDIRECT_ARGUMENT and UNORDERED_ACCESS at once.
#define LIGHT_TREE_INDIRECT_SLOT_MERGE      0 // ceil(N/256)
#define LIGHT_TREE_INDIRECT_SLOT_INTERNAL   1 // ceil((N-1)/256)
#define LIGHT_TREE_INDIRECT_SLOT_NODE       2 // ceil((2N-1)/256)
#define LIGHT_TREE_INDIRECT_SLOT_SORT       3 // first of BITONIC_SORT_STAGE_COUNT

// ---------------------------------------------------------------------------
// BitonicSortPass — vxpgBitonicSort.hlsl. Swaps the root signature out from under
// the light tree mid-build, so its numbering is independent of the tree's.
// ---------------------------------------------------------------------------
#define BITONIC_REG_CB          0 // b
#define BITONIC_REG_SORT_BUFFER 0 // u
#define BITONIC_REG_COUNTER     1 // u
// Ladder length of the 65536 network: 1 presort + 15 outer + 5 inner. The light
// tree's encode kernel writes one DISPATCH argument triple per stage in this
// order, and BitonicSortPass::Sort issues them in the same order — the two loops
// must stay identical or a stage reads another stage's group count.
#define BITONIC_SORT_STAGE_COUNT 21
#define LIGHT_TREE_INDIRECT_SLOT_COUNT (LIGHT_TREE_INDIRECT_SLOT_SORT + BITONIC_SORT_STAGE_COUNT)

// ---------------------------------------------------------------------------
// SuperpixelBuildPass — superpixelBuild.hlsl (private heap table)
// ---------------------------------------------------------------------------
#define SUPERPIXEL_REG_CB              0 // b
#define SUPERPIXEL_REG_INPUT           0 // u
#define SUPERPIXEL_REG_CENTER          1 // u
#define SUPERPIXEL_REG_INDEX           2 // u
#define SUPERPIXEL_REG_SPIXEL_COUNTER  3 // u
#define SUPERPIXEL_REG_SPIXEL_GATHERED 4 // u
#define SUPERPIXEL_REG_FUZZY_WEIGHT    5 // u
#define SUPERPIXEL_REG_FUZZY_INDEX     6 // u

// ---------------------------------------------------------------------------
// LightInjectionPass — lightInjection.hlsl (frame layout plus these)
// ---------------------------------------------------------------------------
#define INJECT_REG_IRRADIANCE           0 // u
#define INJECT_REG_VPL_COUNT            1 // u
#define INJECT_REG_SHADING_POINTS       2 // u
#define INJECT_REG_VOXEL_REPRESENTATIVE 3 // u
#define INJECT_REG_VPL_POSITION         4 // u
#define INJECT_REG_VBUFFER              5 // u
// The injected sample, kept in a form the guided integrator can reuse as its own
// BSDF MIS sample within the same frame (vxpg.injection.reuseInMis). Radiance is
// the SHADED direct light leaving x2 toward x1; the emitter pair is x2's own
// emission and the NEE pdf toward it, which need a different MIS weight and so
// cannot be folded into the first.
#define INJECT_REG_VPL_RADIANCE         6 // u
#define INJECT_REG_VPL_EMITTER          7 // u

// ---------------------------------------------------------------------------
// VBufferPass — vbufferPass.hlsl (frame layout plus this)
// ---------------------------------------------------------------------------
#define VBUFFER_REG_VBUFFER 0 // u

// ---------------------------------------------------------------------------
// GuidedPathTracingPass — guidedPathTracing.hlsl, which
// share one root signature (the adaptive-q update chains onto the raygen), so
// u22/u23 below are the same registers in both shaders.
// ---------------------------------------------------------------------------
#define GUIDED_REG_IRRADIANCE           0  // u
#define GUIDED_REG_VPL_COUNT            1  // u
#define GUIDED_REG_COUNTERS             2  // u
#define GUIDED_REG_COMPACT_IDS          3  // u
#define GUIDED_REG_SUPERPIXEL_INDEX     4  // u
#define GUIDED_REG_INVERSE_INDEX        5  // u
#define GUIDED_REG_VOXEL_REPRESENTATIVE 6  // u
#define GUIDED_REG_VPL_POSITION         7  // u
#define GUIDED_REG_VBUFFER              8  // u
#define GUIDED_REG_FINGERPRINTS         9  // u
#define GUIDED_REG_CLUSTER_ASSIGNMENTS  10 // u
#define GUIDED_REG_CLUSTER_SEEDS        11 // u
#define GUIDED_REG_VISIBILITY_MASK      12 // u
#define GUIDED_REG_LIGHT_TREE_NODES     13 // u
#define GUIDED_REG_COMPACT_TO_LEAF      14 // u
#define GUIDED_REG_CLUSTER_ROOTS        15 // u
#define GUIDED_REG_IMPORTANCE_HEAP      16 // u
#define GUIDED_REG_LIVE_BOUND_MIN       17 // u
#define GUIDED_REG_LIVE_BOUND_MAX       18 // u
#define GUIDED_REG_FUZZY_WEIGHT         19 // u
#define GUIDED_REG_FUZZY_INDEX          20 // u
// Read-only here; written by the injection pass (see INJECT_REG_VPL_RADIANCE).
#define GUIDED_REG_VPL_RADIANCE         21 // u
#define GUIDED_REG_VPL_EMITTER          22 // u
// Forward-chain hand-off: the guide's sampled direction plus its pdf, and the
// voxel span the ray is cut to plus the chain's result code.
#define GUIDED_REG_GUIDE_SAMPLE_DIR     23 // u
#define GUIDED_REG_GUIDE_SAMPLE_SPAN    24 // u
// The chain kernel's shading point; the raygen reconstructs its own from the VBuffer.
#define GUIDED_REG_SHADING_POINTS       25 // u
// Slices in the hand-off, i.e. how many per-pixel samples get their own guide draw. Two covers the
// values this project renders at (benchmarks 1, the settings sweep's best 2); each slice costs two
// screen-sized RGBA32F images, so this is a memory decision, not an algorithmic one.
#define GUIDE_CHAIN_SAMPLE_SLICES       2
// Tile edge for the "swizzle" vendor lever (ADR 0020 R2): launch indices are
// remapped to pixels in Morton order inside a tile of this size, so a wave's
// pixels form a compact block instead of a 32- or 64-wide scanline strip. Must
// be a power of two (the Morton decode assumes it) and shared with the C++ side,
// which pads the dispatch up to whole tiles — without that padding the remap
// stops being a bijection and pixels get dropped or shaded twice.
#define RAYGEN_SWIZZLE_TILE_SHIFT 5
#define RAYGEN_SWIZZLE_TILE_SIZE  (1 << RAYGEN_SWIZZLE_TILE_SHIFT)

// ---------------------------------------------------------------------------
// Rasterization — colorShader.hlsl. Its own layout, not the frame one: it needs
// none of the raytracing frame bindings, so adopting them would only cost root
// DWORDs. b0 is the shared REG_VOXEL_GRID_CB, hence the camera at b1.
// ---------------------------------------------------------------------------
#define RASTER_REG_CAMERA_CB         1 // b
#define RASTER_REG_MODEL_CB          2 // b
#define RASTER_REG_MATERIAL_CB       3 // b
#define RASTER_REG_INDICES           0 // t
#define RASTER_REG_TEXTURES          1 // t
#define RASTER_REG_VOXEL_OCCUPANCY   0 // u
#define RASTER_REG_VOXEL_IRRADIANCE  1 // u
#define RASTER_REG_VOXEL_VPL_COUNT   2 // u
#define RASTER_REG_SHADING_POINTS    3 // u
#define RASTER_REG_SUPERPIXEL_INDEX  4 // u
#define RASTER_REG_SUPERPIXEL_CENTER 5 // u
