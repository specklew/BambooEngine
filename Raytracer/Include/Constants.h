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
        // Bottom light tree leaf ceiling. The node index is uint (SIByL uses uint16_t), so
        // the old 2N-1 <= 65535 constraint is gone, and the sort key now spends 18 bits on
        // the compact voxel id (262144 leaves). What binds instead is the compaction buffer:
        // voxelGuidingBuild.hlsl drops every voxel past VOXEL_GUIDING_CAPACITY, and a leaf
        // cannot exist without a compacted voxel, so raising this means raising that first.
        // The encode kernel clamps the lit-voxel count to this (see ADR 0003), and
        // TreeBuildDispatchArgs::overflowFlag still wants a continuous CPU-side read so
        // truncation does not depend on the hand-armed cluster probe.
        constexpr int LIGHT_TREE_MAX_LEAVES = VOXEL_GUIDING_CAPACITY;
        // Bitonic sort-key buffer capacity. Must be a power of two >= the leaf ceiling. NOT
        // a per-frame cost: the ladder pads the live leaf count up to a power of two and
        // every stage above it dispatches zero groups, so a raised capacity costs 8 B/key of
        // storage and nothing else. BitonicSortPass::kCapacity and BITONIC_SORT_STAGE_COUNT
        // are derived from this number and must move with it.
        constexpr int LIGHT_TREE_SORT_CAPACITY = 131072;
        // VXPG V2 Stage B superpixels (SLIC over the ShadingPoints G-buffer).
        // map_size = ceil(screen / SUPERPIXEL_SIZE); gather cap = SUPERPIXEL_SIZE^2.
        constexpr int SUPERPIXEL_SIZE = 32;
        // 1 = the paper's shipped config ("only execute the pixel assignment
        // stage once", supplemental 1.3). Each extra iteration is one Associate +
        // one SumCenters node: 0.108 + 0.030 ms at 1920x1080 measured 2026-08-23,
        // so 5 iterations would cost ~0.55 ms/frame — for no measured guiding
        // benefit.
        constexpr int SUPERPIXEL_ITERATIONS = 1;
        constexpr int SUPERPIXEL_GATHER_CAP = SUPERPIXEL_SIZE * SUPERPIXEL_SIZE; // 1024
    }
}
