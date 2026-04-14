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
        constexpr int STATIC_SAMPLERS_COUNT = 6;
    }
}
