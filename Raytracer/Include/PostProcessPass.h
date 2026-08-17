#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"
#include "Resources/Texture.h"

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

    // Two graph nodes: the tonemap dispatch, then the copy to the back buffer.
    // Neither places barriers — the graph synthesizes them from the declarations.
    void Dispatch(Texture& input, const PostProcessParams& params = {});
    void CopyToBackBuffer(Texture& backBuffer);

    void OnResize();

    // Returns the post-process output texture (DXGI_FORMAT_R8G8B8A8_UNORM).
    Texture& GetOutputBuffer() const { return *m_outputBuffer; }

private:
    void CreateResources();
    void CreateRootSignature();
    void CreatePSO();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;
    std::unique_ptr<Texture>                           m_outputBuffer;
    RootSignature                                      m_rootSignature;
    ComputeProgram*                                    m_program = nullptr;

    // Which copy of the pass's global-heap block this frame writes and binds.
    uint32_t m_ringSlot = 0;

    bool m_initialized = false;
};
