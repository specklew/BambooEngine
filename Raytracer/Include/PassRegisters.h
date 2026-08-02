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
#ifdef __cplusplus
#pragma once
#endif

#include "FrameBindingRegisters.h"

// Voxel grid constants, shared by every pass that reads the grid: the guided
// integrator, light injection and the rasterization debug views.
#define REG_VOXEL_GRID_CB 4

// ---------------------------------------------------------------------------
// FrameAccumulationPass — accumulation.hlsl
// ---------------------------------------------------------------------------
#define ACCUM_REG_CURRENT 0 // t
#define ACCUM_REG_ACCUM   0 // u
#define ACCUM_REG_DISPLAY 1 // u
#define ACCUM_REG_CB      0 // b

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

// ---------------------------------------------------------------------------
// BitonicSortPass — vxpgBitonicSort.hlsl. Swaps the root signature out from under
// the light tree mid-build, so its numbering is independent of the tree's.
// ---------------------------------------------------------------------------
#define BITONIC_REG_CB          0 // b
#define BITONIC_REG_SORT_BUFFER 0 // u
#define BITONIC_REG_COUNTER     1 // u

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
#define INJECT_REG_IRRADIANCE           1 // u
#define INJECT_REG_VPL_COUNT            2 // u
#define INJECT_REG_SHADING_POINTS       3 // u
#define INJECT_REG_VOXEL_REPRESENTATIVE 4 // u
#define INJECT_REG_VPL_POSITION         5 // u
#define INJECT_REG_VBUFFER              6 // u

// ---------------------------------------------------------------------------
// VBufferPass — vbufferPass.hlsl (frame layout plus this)
// ---------------------------------------------------------------------------
#define VBUFFER_REG_VBUFFER 9 // u

// ---------------------------------------------------------------------------
// GuidedPathTracingPass — guidedPathTracing.hlsl and vxpgAdaptiveQ.hlsl, which
// share one root signature (the adaptive-q update chains onto the raygen), so
// u22/u23 below are the same registers in both shaders.
// ---------------------------------------------------------------------------
#define GUIDED_REG_IRRADIANCE           1  // u
#define GUIDED_REG_VPL_COUNT            2  // u
#define GUIDED_REG_COUNTERS             3  // u
#define GUIDED_REG_COMPACT_IDS          4  // u
#define GUIDED_REG_SUPERPIXEL_INDEX     5  // u
#define GUIDED_REG_INVERSE_INDEX        6  // u
#define GUIDED_REG_VOXEL_REPRESENTATIVE 7  // u
#define GUIDED_REG_VPL_POSITION         8  // u
#define GUIDED_REG_VBUFFER              9  // u
#define GUIDED_REG_FINGERPRINTS         10 // u
#define GUIDED_REG_CLUSTER_ASSIGNMENTS  11 // u
#define GUIDED_REG_CLUSTER_SEEDS        12 // u
#define GUIDED_REG_VISIBILITY_MASK      13 // u
#define GUIDED_REG_LIGHT_TREE_NODES     14 // u
#define GUIDED_REG_COMPACT_TO_LEAF      15 // u
#define GUIDED_REG_CLUSTER_ROOTS        16 // u
#define GUIDED_REG_IMPORTANCE_HEAP      17 // u
#define GUIDED_REG_LIVE_BOUND_MIN       18 // u
#define GUIDED_REG_LIVE_BOUND_MAX       19 // u
#define GUIDED_REG_FUZZY_WEIGHT         20 // u
#define GUIDED_REG_FUZZY_INDEX          21 // u
#define GUIDED_REG_TILE_GUIDE_Q         22 // u
#define GUIDED_REG_TILE_STRATEGY_STATS  23 // u

// ---------------------------------------------------------------------------
// Rasterization — colorShader.hlsl. Its own layout, not the frame one; raster
// joins the frame layout when it becomes a graph node (ADR 0017 phase 5).
// ---------------------------------------------------------------------------
#define RASTER_REG_CAMERA_CB        0 // b
#define RASTER_REG_MODEL_CB         1 // b
#define RASTER_REG_MATERIAL_CB      2 // b
#define RASTER_REG_INDICES          2 // t
#define RASTER_REG_TEXTURES         3 // t
#define RASTER_REG_VOXEL_OCCUPANCY  1 // u
#define RASTER_REG_VOXEL_IRRADIANCE 2 // u
#define RASTER_REG_VOXEL_VPL_COUNT  3 // u
#define RASTER_REG_SHADING_POINTS   4 // u
#define RASTER_REG_SUPERPIXEL_INDEX 7 // u
#define RASTER_REG_SUPERPIXEL_CENTER 8 // u
