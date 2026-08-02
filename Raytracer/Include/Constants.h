#pragma once

#include "FrameBindingRegisters.h"

namespace Constants
{
    constexpr int MAX_STRING_LEN = 256;

    namespace Graphics
{
        constexpr int NUM_FRAMES = 3;
        // Shared with the shaders, which size their bindless array from the same define.
        constexpr int MAX_TEXTURES = FRAME_MAX_TEXTURES;
        constexpr int MAX_OBJECTS = 64;
        // Descriptor slot layout lives in GlobalDescriptorHeap.h (ADR 0017 L2).
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
        // 1 = the paper's shipped config ("only execute the pixel assignment
        // stage once", supplemental 1.3); 5 iterations with live center updates
        // cost ~1.3 ms/frame at 1080p for no measured guiding benefit.
        constexpr int SUPERPIXEL_ITERATIONS = 1;
        constexpr int SUPERPIXEL_GATHER_CAP = SUPERPIXEL_SIZE * SUPERPIXEL_SIZE; // 1024
    }
}
