#pragma once
#include "DxrPass.h"
#include "RenderTechnique.h"

// A user-selectable raytracing integrator: a DXR dispatch that owns a
// full-screen output image and contributes it to the frame's graph. The
// renderer's display chain (accumulate -> tonemap -> present copy) consumes
// whatever BuildGraph hands back, so a technique never touches the back buffer.
class DxrTechnique : public DxrPass, public RenderTechnique
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<Scene> initialScene,
        std::shared_ptr<PassConstants> passConstants) override;

    GraphResourceHandle BuildGraph(RenderGraph& graph, const FrameGraphContext& frame) override;

    void OnResize() override;
    void OnShaderReload() override { DxrPass::OnShaderReload(); }
    void OnSceneChange(std::shared_ptr<Scene> scene) override { DxrPass::OnSceneChange(scene); }

    bool SetShaderVariantKey(const std::string& key) override;
    std::string GetShaderVariantKey() const override { return m_shaderVariantKey; }

    int  GetDebugMode() const override;
    bool HasActiveDebugView() const override;

    std::vector<DebugView> GetDebugViews() const override;
    bool SetDebugView(int index) override;

    Texture* GetOutputTexture() const override { return m_outputResource.get(); }

protected:
    // What the dispatch touches beyond the frame layout, declared into the node
    // that runs it, and the nodes that run after it — the second half is separate
    // because a node cannot be added while another node's declaration is open.
    // A technique that needs neither places no barriers at all.
    virtual void DeclareDispatchResources(RenderGraph& graph, RenderGraphPassBuilder& dispatchPass) {}
    virtual void AppendPostDispatchNodes(RenderGraph& graph) {}

    void CreateShaderResourceHeap() override;

    // This frame's VXPG imports, valid only for the duration of BuildGraph.
    const VxpgGraphHandles* m_frameGuiding = nullptr;

private:
    void CreateOutputTexture();

    std::unique_ptr<Texture> m_outputResource;
};
