#pragma once
#include "CommandContext.h"

#include "BufferDebugView.h"
#include "RenderGraph.h"
#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"
#include "Resources/Texture.h"
#include "VxpgGraphHandles.h"

class VoxelGuidingBuildPass;
class VoxelizationPass;
class VxpgClusterPass;

// Paints one buffer-debug view from the VXPG products (ADR 0017 phase 5b). It is
// a node rather than a branch inside three shaders, so the same view renders
// whichever technique is active — including none of them, since the integrator
// is skipped entirely while a buffer view is up.
class DebugViewPass
{
public:
    void Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);
    void SetVoxelizationPass(const std::shared_ptr<VoxelizationPass>& voxelPass) { m_voxelPass = voxelPass; }
    // Producers of the cluster view's buffers; without both, that view has
    // nothing to read and is not offered.
    void SetGuidingPasses(const std::shared_ptr<VoxelGuidingBuildPass>& buildPass,
                          const std::shared_ptr<VxpgClusterPass>& clusterPass)
    {
        m_buildPass   = buildPass;
        m_clusterPass = clusterPass;
    }

    void OnResize();

    static bool IsActive() { return g_bufferDebugView.Get() != BufferDebugView::None; }

    // Only what the active view samples, so culling keeps exactly the stages
    // behind it alive and drops the rest.
    void DeclareGraphResources(RenderGraphPassBuilder& pass, GraphResourceHandle output,
                               const VxpgGraphHandles& vxpg) const;

    void Dispatch();
    void CopyToBackBuffer(Texture& backBuffer);

    Texture& GetOutputBuffer() const { return *m_outputBuffer; }

private:
    void CreateResources();
    void CreateRootSignature();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;
    std::unique_ptr<Texture>                           m_outputBuffer;
    RootSignature                                      m_rootSignature;
    ComputeProgram*                                    m_program = nullptr;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass>             m_buildPass;
    std::shared_ptr<VxpgClusterPass>                   m_clusterPass;
};
