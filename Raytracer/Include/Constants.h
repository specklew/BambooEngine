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
        constexpr int STATIC_SAMPLERS_COUNT = 6;
        // Max voxels in the compacted guiding distribution (matches SIByL VXGuider_MAX_CAPACITY)
        constexpr int VOXEL_GUIDING_CAPACITY = 131072;
    }
}
