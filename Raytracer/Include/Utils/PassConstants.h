#pragma once
#include "Utils/FrameConstantRing.h"

class PassConstants
{
public:

    //TODO: Maybe I could somehow make this more flexible for new params?

    PassConstants();

    // Publishes `data` into the frame's own copy of the buffer, and is what
    // GetGpuVirtualAddress() answers for until the next frame calls it. Called
    // once per frame from Renderer::Update, before any pass binds.
    void Map(uint32_t frameIndex);
    D3D12_GPU_VIRTUAL_ADDRESS GetGpuVirtualAddress() const;

    struct MappedData
    {
        float uvCoordX = 0.0f;
        float uvCoordY = 0.0f;
        int debugMode = 0;
        uint32_t numLights = 0;
        uint32_t numSamplesPerPixel = 4;
        uint32_t numBounces = 1;
        uint32_t frameIndex = 0;
        uint32_t guidingFlags = 0; // bit 0 = power MIS; doubles as the pad HLSL inserts before float3
        DirectX::XMFLOAT3 cameraWorldPos = {0.0f, 0.0f, 0.0f};
        // 1 = per-pixel sub-pixel jitter on the shared VBuffer primaries (the
        // jitter is derived per pixel in-shader from pixel + frameIndex).
        uint32_t vbufferJitterEnabled = 0;
        // Indirect skybox radiance clamp (firefly suppression for benchmark
        // convergence). 0 = disabled/unbiased. See passConstants.hlsl.
        float indirectSkyClamp = 0.0f;
        // 1 = skybox radiance lights surfaces via indirect rays (default);
        // 0 = sky stays visible as background but contributes no lighting.
        // See passConstants.hlsl.
        uint32_t skyLightingEnabled = 1;
        uint32_t lightPoolCount = 0;
        float lightPoolTotalPower = 0.0f;
    } data;

private:
    FrameConstantRing m_ring;
    uint32_t          m_frameIndex = 0;
};
