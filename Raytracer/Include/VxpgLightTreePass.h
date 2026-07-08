#pragma once

#include "Resources/RWStructuredBuffer.h"
#include "BitonicSortPass.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgClusterPass;

// VXPG bottom light tree: a Karras LBVH over the lit voxels (SIByL vxguiding
// tree-encode -> bitonic sort -> tree-initial -> tree-internal -> tree-merge).
// Runs each frame after clustering. Produces the node array + per-cluster root
// nodes + the compact->leaf reverse map the guided integrator will sample.
// All buffers are root UAV/SRV (no global-heap slots). Owns its BitonicSortPass.
class VxpgLightTreePass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelizationPass>                  voxelPass,
        std::shared_ptr<VoxelGuidingBuildPass>             buildPass,
        std::shared_ptr<VxpgClusterPass>                   clusterPass);

    void Run();

    // GPU-only opaque node record; >= the HLSL LightTreeNode structured-buffer
    // stride (32 B). Never read on the CPU — sized generously so the shader's
    // own stride can't run the buffer out of bounds.
    struct LightTreeNodeGpu { uint32_t opaque[12]; };

    // Mirror of the shader TreeBuildDispatchArgs (numValidVoxels at byte 12).
    struct TreeBuildDispatchArgsGpu
    {
        int32_t  dispatchLeaf[3];
        uint32_t numValidVoxels;
        int32_t  dispatchInternal[3];
        uint32_t overflowFlag;
        int32_t  dispatchNode[3];
        uint32_t padding1;
        int32_t  drawRects[4];
    };

    // Consumed by the guided integrator (later) + debug view 11.
    D3D12_GPU_VIRTUAL_ADDRESS GetNodesBufferVA() const { return m_nodes->GetGPUVirtualAddress(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetCompactToLeafBufferVA() const { return m_compactToLeaf->GetGPUVirtualAddress(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetClusterRootsBufferVA() const { return m_clusterRoots->GetGPUVirtualAddress(); }

private:
    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass>             m_buildPass;
    std::shared_ptr<VxpgClusterPass>                   m_clusterPass;

    BitonicSortPass m_sort;

    std::unique_ptr<RWStructuredBuffer<uint64_t>>                 m_sortKeys;      // SIByL u_Codes
    std::unique_ptr<RWStructuredBuffer<LightTreeNodeGpu>>         m_nodes;         // SIByL u_Nodes
    std::unique_ptr<RWStructuredBuffer<uint32_t>>                 m_leafRanges;    // SIByL u_Descendant (dead)
    std::unique_ptr<RWStructuredBuffer<int32_t>>                  m_compactToLeaf; // SIByL compact2leaf
    std::unique_ptr<RWStructuredBuffer<int32_t>>                  m_clusterRoots;  // SIByL cluster_roots
    std::unique_ptr<RWStructuredBuffer<TreeBuildDispatchArgsGpu>> m_dispatchArgs;  // SIByL u_ConstrIndirectArgs
    std::unique_ptr<RWStructuredBuffer<uint32_t>>                 m_nodeVisited;   // merge sibling-gate (own scalar buffer)

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_encodePso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_initialPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_internalPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_mergePso;

    bool m_initialized = false;
};
