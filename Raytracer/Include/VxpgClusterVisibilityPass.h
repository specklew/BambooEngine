#pragma once

#include "Resources/RWStructuredBuffer.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgClusterPass;
class SuperpixelBuildPass;
class Scene;

// VXPG cluster-visibility pass (MRCS "C-lean" soft visibility): fills the
// per-superpixel x per-cluster visibility matrix (hard 32-bit mask + soft
// BRDF-weighted avg-visibility) that makes the guide view-adaptive. Three
// compute kernels run each frame after the superpixel + cluster passes:
//   ClearClusterVisibility   -> zero mask / counts / avg
//   GatherClusterLightPoints -> file each pixel's VPL into its cluster drawer
//                               and seed the mask bit for proven connections
//   CheckClusterVisibility   -> 32 shadow-ray probes per (superpixel, cluster),
//                               Cook-Torrance-weighted; needs SM 6.6 (inline
//                               RayQuery + [WaveSize(32)] wave reductions).
// Reuses the global SRV/CBV/UAV heap for the scene binding (camera, TLAS,
// geometry, textures, VBuffer, superpixel textures) exactly like the guided
// integrator; owns the cluster drawers, counts, mask, and avg buffers.
class VxpgClusterVisibilityPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>       globalHeap,
        std::shared_ptr<VoxelizationPass>                  voxelPass,
        std::shared_ptr<VoxelGuidingBuildPass>             buildPass,
        std::shared_ptr<VxpgClusterPass>                   clusterPass,
        std::shared_ptr<SuperpixelBuildPass>               superpixelPass);

    void SetScene(const std::shared_ptr<Scene>& scene) { m_scene = scene; }

    void OnResize(uint32_t width, uint32_t height);

    void Run(uint32_t frameIndex);

    // Mask read by guided PT debug view 10 (Renderer writes its global-heap UAV).
    ID3D12Resource* GetMaskResource() const { return m_mask.Get(); }
    RWStructuredBuffer<float>* GetAvgVisibilityBuffer() const { return m_avgVisibility.get(); }

private:
    void CreateFixedBuffers();
    void CreateResolutionBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>       m_globalHeap;

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass> m_buildPass;
    std::shared_ptr<VxpgClusterPass>       m_clusterPass;
    std::shared_ptr<SuperpixelBuildPass>   m_superpixelPass;
    std::shared_ptr<Scene>                 m_scene;

    std::unique_ptr<RWStructuredBuffer<DirectX::XMFLOAT4>> m_clusterGatheredLightPoints; // 32 x 1024
    std::unique_ptr<RWStructuredBuffer<uint32_t>>          m_clusterLightPointCounts;    // 32
    std::unique_ptr<RWStructuredBuffer<float>>             m_avgVisibility;              // mapX*mapY*32
    Microsoft::WRL::ComPtr<ID3D12Resource>                m_mask;                        // (mapX, mapY) R32_UINT

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_gatherPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_checkPso;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_mapX = 0;
    uint32_t m_mapY = 0;
    bool     m_initialized = false;
};
