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
        DirectX::XMFLOAT3 cameraWorldPos = {0.0f, 0.0f, 0.0f};
    } data;

private:
    std::unique_ptr<ConstantBuffer> m_buffer;
    MappedData m_mappedData;
};
