#pragma once
#include "CommandContext.h"
#include "RenderTechnique.h"
#include "RootSignatureLibrary.h"

// Traditional vertex/pixel pipeline over the scene's meshes, drawing straight
// into the back buffer. A peer of the raytracing integrators rather than a mode
// the renderer switches into: it registers, it declares nodes, and it is culled
// or kept on the same terms as anything else in the graph.
class RasterizationTechnique : public RenderTechnique
{
public:
    // Must be called before Initialize — the pipeline state bakes the formats in.
    void SetFrameTargetFormats(DXGI_FORMAT backBufferFormat, DXGI_FORMAT depthStencilFormat);

    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<Scene> initialScene,
        std::shared_ptr<PassConstants> passConstants) override;

    GraphResourceHandle BuildGraph(RenderGraph& graph, const FrameGraphContext& frame) override;

    // The debug views are the only raster consumer of the VXPG subgraph.
    bool UsesVoxelGuiding() const override;
    int  GetDebugMode() const override;

    std::vector<DebugView> GetDebugViews() const override;
    bool SetDebugView(int index) override;

    void OnShaderReload() override { CreatePipelineState(); }
    void OnSceneChange(std::shared_ptr<Scene> scene) override { m_scene = std::move(scene); }

private:
    void CreateRootSignature();
    void CreatePipelineState();

    void Clear(const FrameGraphContext& frame) const;
    void DrawScene(const FrameGraphContext& frame) const;

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;

    RootSignature                               m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineStateObject;
    Microsoft::WRL::ComPtr<IDxcBlob>            m_vertexShader;
    Microsoft::WRL::ComPtr<IDxcBlob>            m_pixelShader;

    DXGI_FORMAT m_backBufferFormat   = DXGI_FORMAT_R8G8B8A8_UNORM;
    DXGI_FORMAT m_depthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;

    std::shared_ptr<Scene>         m_scene;
    std::shared_ptr<PassConstants> m_passConstants;
};
