#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"

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
        std::shared_ptr<VoxelizationPass>                  voxelPass,
        std::shared_ptr<VoxelGuidingBuildPass>             buildPass,
        std::shared_ptr<VxpgClusterPass>                   clusterPass,
        std::shared_ptr<SuperpixelBuildPass>               superpixelPass);

    void SetScene(const std::shared_ptr<Scene>& scene) { m_scene = scene; }

    void OnResize(uint32_t width, uint32_t height);

    // One graph node per kernel: the clear -> gather -> check ordering comes from
    // the declarations rather than hand-placed barriers.
    void RunClear(uint32_t frameIndex);
    void RunGather(uint32_t frameIndex);
    void RunCheck(uint32_t frameIndex);

    // Mask read by guided PT debug view 10 (Renderer writes its global-heap UAV).
    ID3D12Resource* GetMaskResource() const { return m_mask.Get(); }
    RWStructuredBuffer<float>* GetAvgVisibilityBuffer() const { return m_avgVisibility.get(); }
    RWStructuredBuffer<DirectX::XMFLOAT4>* GetClusterGatheredLightPointsBuffer() const { return m_clusterGatheredLightPoints.get(); }
    RWStructuredBuffer<uint32_t>* GetClusterLightPointCountsBuffer() const { return m_clusterLightPointCounts.get(); }

private:
    void CreateFixedBuffers();
    void CreateResolutionBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    // Every kernel re-binds: separate nodes may have barriers placed between
    // them, so none may inherit another's root state. False = cannot run.
    bool BindCommon(uint32_t frameIndex);

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;

    std::shared_ptr<VoxelizationPass>      m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass> m_buildPass;
    std::shared_ptr<VxpgClusterPass>       m_clusterPass;
    std::shared_ptr<SuperpixelBuildPass>   m_superpixelPass;
    std::shared_ptr<Scene>                 m_scene;

    std::unique_ptr<RWStructuredBuffer<DirectX::XMFLOAT4>> m_clusterGatheredLightPoints; // 32 x 1024
    std::unique_ptr<RWStructuredBuffer<uint32_t>>          m_clusterLightPointCounts;    // 32
    std::unique_ptr<RWStructuredBuffer<float>>             m_avgVisibility;              // mapX*mapY*32
    Microsoft::WRL::ComPtr<ID3D12Resource>                m_mask;                        // (mapX, mapY) R32_UINT

    RootSignature m_rootSig;
    ComputeProgram* m_clearProgram = nullptr;
    ComputeProgram* m_gatherProgram = nullptr;
    ComputeProgram* m_checkProgram = nullptr;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_mapX = 0;
    uint32_t m_mapY = 0;
    bool     m_initialized = false;
};
