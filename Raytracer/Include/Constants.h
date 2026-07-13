#pragma once

namespace Constants
{
    constexpr int MAX_STRING_LEN = 256;
    
    namespace Graphics
{
        constexpr int NUM_FRAMES = 3;
        constexpr int MAX_TEXTURES = 512;
        constexpr int MAX_OBJECTS = 64;
        constexpr int NUM_BASE_DESCRIPTORS = 6;
        constexpr int SKYBOX_DESCRIPTOR_INDEX = NUM_BASE_DESCRIPTORS + MAX_TEXTURES; // 518
        constexpr int VOXEL_OCCUPANCY_DESCRIPTOR_INDEX = SKYBOX_DESCRIPTOR_INDEX + 1; // 519
        constexpr int VOXEL_IRRADIANCE_DESCRIPTOR_INDEX = VOXEL_OCCUPANCY_DESCRIPTOR_INDEX + 1; // 520
        constexpr int VOXEL_VPL_COUNT_DESCRIPTOR_INDEX = VOXEL_IRRADIANCE_DESCRIPTOR_INDEX + 1; // 521
        // VXPG V2: ShadingPoints G-buffer UAV (primary worldPos + octa normal), written by light injection.
        constexpr int SHADINGPOINTS_DESCRIPTOR_INDEX = VOXEL_VPL_COUNT_DESCRIPTOR_INDEX + 1; // 522
        // VXPG V2 Stage B: superpixel index + representative center UAVs (debug views 15/16, raster table u7/u8).
        constexpr int SUPERPIXEL_INDEX_DESCRIPTOR_INDEX = SHADINGPOINTS_DESCRIPTOR_INDEX + 1; // 523
        constexpr int SUPERPIXEL_CENTER_DESCRIPTOR_INDEX = SUPERPIXEL_INDEX_DESCRIPTOR_INDEX + 1; // 524
        // VXPG faithful port (B+): per-voxel representative VPL (pos + octa normal, Texture3D)
        // and per-pixel VPL hit position (screen, Texture2D), both written by light injection.
        constexpr int VOXEL_REPRESENTATIVE_DESCRIPTOR_INDEX = SUPERPIXEL_CENTER_DESCRIPTOR_INDEX + 1; // 525
        constexpr int VPL_POSITION_DESCRIPTOR_INDEX = VOXEL_REPRESENTATIVE_DESCRIPTOR_INDEX + 1; // 526
        // Shared primary-visibility buffer (ADR 0004), written by VBufferPass,
        // consumed by light injection + the guided integrator.
        constexpr int VBUFFER_DESCRIPTOR_INDEX = VPL_POSITION_DESCRIPTOR_INDEX + 1; // 527
        // VXPG cluster-visibility (cvis): per-superpixel pixel-list + counter (from
        // the superpixel pass) and the cluster-visibility mask (written by cvis),
        // read by the cvis check kernel + guided PT debug view 10.
        constexpr int SPIXEL_GATHERED_DESCRIPTOR_INDEX = VBUFFER_DESCRIPTOR_INDEX + 1; // 528
        constexpr int SPIXEL_COUNTER_DESCRIPTOR_INDEX = SPIXEL_GATHERED_DESCRIPTOR_INDEX + 1; // 529
        constexpr int CLUSTER_VISIBILITY_MASK_DESCRIPTOR_INDEX = SPIXEL_COUNTER_DESCRIPTOR_INDEX + 1; // 530
        // Fuzzy 4-nearest superpixel blend (SIByL u_fuzzyWeight / u_fuzzyIdx):
        // per-pixel 4 nearest superpixel centers + 1/dist^2 weights, written by
        // the superpixel pass, consumed by the guided integrator's mixture pdf.
        constexpr int FUZZY_WEIGHT_DESCRIPTOR_INDEX = CLUSTER_VISIBILITY_MASK_DESCRIPTOR_INDEX + 1; // 531
        constexpr int FUZZY_INDEX_DESCRIPTOR_INDEX = FUZZY_WEIGHT_DESCRIPTOR_INDEX + 1; // 532
        constexpr int STATIC_SAMPLERS_COUNT = 6;
        // Max voxels in the compacted guiding distribution (matches SIByL VXGuider_MAX_CAPACITY)
        constexpr int VOXEL_GUIDING_CAPACITY = 131072;
        // Bottom light tree: the uint16 node-index ceiling. The node array holds
        // 2N-1 entries and every index must fit uint16, so 2N-1 <= 65535 => N <=
        // 32768. The encode kernel clamps the lit-voxel count to this (see ADR 0003).
        constexpr int LIGHT_TREE_MAX_LEAVES = 32768;
        // Bitonic sort-key buffer capacity (SIByL element_count = 65536).
        constexpr int LIGHT_TREE_SORT_CAPACITY = 65536;
        // VXPG V2 supervoxels: coarse grid cell = voxelCoord / clusterFactor. The
        // factor is a FLOOR (SUPERVOXEL_GRID_FACTOR) that adapts upward so svDim
        // never exceeds SUPERVOXEL_DIM_CAP, keeping the supervoxel count within
        // MAX_SUPERVOXELS for any grid resolution (the Stage C matrix-width budget).
        // SUPERVOXEL_DIM_CAP = cbrt(MAX_SUPERVOXELS): 8^3 = 512.
        constexpr int SUPERVOXEL_GRID_FACTOR = 16;
        constexpr int MAX_SUPERVOXELS = 512;
        constexpr int SUPERVOXEL_DIM_CAP = 8;
        // VXPG V2 Stage B superpixels (SLIC over the ShadingPoints G-buffer).
        // map_size = ceil(screen / SUPERPIXEL_SIZE); gather cap = SUPERPIXEL_SIZE^2.
        constexpr int SUPERPIXEL_SIZE = 32;
        constexpr int SUPERPIXEL_ITERATIONS = 5;
        constexpr int SUPERPIXEL_GATHER_CAP = SUPERPIXEL_SIZE * SUPERPIXEL_SIZE; // 1024
    }
}
