#pragma once

class ConstantBuffer;

class PassConstants
{
public:

    //TODO: Maybe I could somehow make this more flexible for new params?
    
    PassConstants();
    void Map();
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
    } data;

private:
    std::unique_ptr<ConstantBuffer> m_buffer;
    MappedData m_mappedData;
};
