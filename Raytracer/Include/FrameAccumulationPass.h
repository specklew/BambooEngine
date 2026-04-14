#pragma once

class FrameAccumulationPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    void Render(
        const Microsoft::WRL::ComPtr<ID3D12Resource>& currentFrameOutput);

    void Update(double elapsedTime);
    void Reset();
    void OnResize();

    uint32_t GetFrameCount()    const { return m_frameCount; }
    double   GetAccumulatedTime() const { return m_accumulatedTime; }
    uint32_t GetResetCount()    const { return m_resetCount; }
    const Microsoft::WRL::ComPtr<ID3D12Resource>& GetDisplayBuffer() const { return m_displayBuffer; }

private:
    void CreateResources();
    void CreateRootSignature();
    void CreatePSO();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_accumulationBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_displayBuffer;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>        m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState>        m_pso;
    Microsoft::WRL::ComPtr<IDxcBlob>                   m_computeShaderBlob;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>       m_descriptorHeap;  // For input texture SRV

    uint32_t m_frameCount      = 0;
    double   m_accumulatedTime = 0.0;
    uint32_t m_resetCount      = 0;
    bool     m_initialized     = false;
};
