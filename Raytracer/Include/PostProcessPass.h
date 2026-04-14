#pragma once

struct PostProcessParams
{
    float exposure   = 1.0f;
    float contrast   = 1.0f;
    float saturation = 1.0f;
    float lift       = 0.0f;
};

class PostProcessPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    void Render(
        const Microsoft::WRL::ComPtr<ID3D12Resource>& input,
        const Microsoft::WRL::ComPtr<ID3D12Resource>& backBuffer,
        const PostProcessParams& params = {});

    void OnResize();

    // Returns the post-process output texture (DXGI_FORMAT_R8G8B8A8_UNORM).
    // In D3D12_RESOURCE_STATE_COPY_SOURCE after Render().
    const Microsoft::WRL::ComPtr<ID3D12Resource>& GetOutputBuffer() const { return m_outputBuffer; }

private:
    void CreateResources();
    void CreateRootSignature();
    void CreatePSO();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_outputBuffer;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>        m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState>        m_pso;
    Microsoft::WRL::ComPtr<IDxcBlob>                   m_computeShaderBlob;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>       m_descriptorHeap;

    bool m_initialized = false;
};
