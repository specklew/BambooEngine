#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"

#include "Resources/RWStructuredBuffer.h"
#include "BitonicSortPass.h"
#include "Utils/CVars.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgClusterPass;
class VxpgClusterVisibilityPass;
class GpuMemoryReport;

// How the top-level (per-superpixel) tree weights the 32 clusters. The first two
// are SIByL's own pair and are both visibility-aware; SIByL ships Average.
// IntensityOnly is the research plan's "power-proportional" strategy: the weight
// is the cluster's intensity alone, visibility fixed at 1. It has no other
// consumer, so selecting it also culls the whole cluster-visibility stage from
// the frame (see Renderer::BuildVxpgGraph) — a strategy that ignores a signal
// must not be charged for producing it.
// AverageVisibility measured BEST of the two SIByL modes once the cvis facing-gate
// flip landed (2026-07-10: FLIP 0.01476 vs BinaryVisibility 0.01526 vs pre-fix
// Average 0.01552 on the Standard Look b1 benchmark) — the raw-normal gate was
// what starved Average's probe pairs on back-wound surfaces; the binary 0/1
// weighting over-samples barely-visible clusters and is strictly worse post-fix.
enum class TopLevelImportance : int
{
    BinaryVisibility  = 0,
    AverageVisibility = 1,
    IntensityOnly     = 2,
};

// Lives beside its enum because both the light tree pass (which packs it into the
// top-level root constants) and the Renderer (which decides whether the
// cluster-visibility stage is needed at all) read the same object.
inline AutoCVarEnum g_topLevelImportance("vxpg.topLevelTree.importance",
    "Top-level cluster weighting: visibility mask, soft average visibility (SIByL), or intensity alone",
    TopLevelImportance::AverageVisibility, CVarFlags::None,
    {"cluster intensity zeroed unless the superpixel's mask bit is set",
     "cluster intensity times the soft BRDF-weighted visible fraction (SIByL default)",
     "cluster intensity alone, visibility ignored - skips the cluster-visibility stage"});

// VXPG bottom light tree: a Karras LBVH over the lit voxels (SIByL vxguiding
// tree-encode -> bitonic sort -> tree-initial -> tree-internal -> tree-merge).
// Runs each frame after clustering. Produces the node array + per-cluster root
// nodes + the compact->leaf reverse map the guided integrator will sample.
// All buffers are root UAV/SRV (no global-heap slots). Owns its BitonicSortPass.
class VxpgLightTreePass
{
public:
    // P5: the resources this stage holds, for the guiding chain's memory report.
    // Declared per pass rather than collected from a base class because ADR 0017 left
    // most of this chain as raw ComPtr, which no base class ever sees.
    void ReportMemory(GpuMemoryReport& report) const;
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelizationPass>                  voxelPass,
        std::shared_ptr<VoxelGuidingBuildPass>             buildPass,
        std::shared_ptr<VxpgClusterPass>                   clusterPass,
        std::shared_ptr<VxpgClusterVisibilityPass>         clusterVisibilityPass);

    // Sizes the per-superpixel importance heap (mapX*mapY*64 floats).
    void OnResize(uint32_t width, uint32_t height);

    // One graph node per stage: the ordering between them comes from the
    // declarations rather than hand-placed barriers.
    void RunClear();
    void RunEncode();
    void RunSort();
    void RunInitial();
    void RunInternal();
    void RunMerge();
    void RunTopLevel();

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

    // One DISPATCH argument triple. The encode kernel fills the whole array from
    // the live leaf count (three tree stages, then the bitonic ladder), so a frame
    // that lit 1500 voxels stops paying the 32768-leaf / 65536-key worst case.
    struct DispatchArgsGpu { uint32_t threadGroupCount[3]; };

    // Consumed by the guided integrator (later) + debug view 11.
    D3D12_GPU_VIRTUAL_ADDRESS GetNodesBufferVA() const { return m_nodes->GetGPUVirtualAddress(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetCompactToLeafBufferVA() const { return m_compactToLeaf->GetGPUVirtualAddress(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetClusterRootsBufferVA() const { return m_clusterRoots->GetGPUVirtualAddress(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetSuperpixelClusterHeapBufferVA() const { return m_spixelClusterHeap->GetGPUVirtualAddress(); }

    // Same buffers as the VAs above, for the render graph's read/write declarations.
    RWStructuredBuffer<LightTreeNodeGpu>* GetNodesBuffer() const { return m_nodes.get(); }
    RWStructuredBuffer<int32_t>* GetCompactToLeafBuffer() const { return m_compactToLeaf.get(); }
    RWStructuredBuffer<int32_t>* GetClusterRootsBuffer() const { return m_clusterRoots.get(); }
    RWStructuredBuffer<float>* GetSuperpixelClusterHeapBuffer() const { return m_spixelClusterHeap.get(); }
    RWStructuredBuffer<uint64_t>* GetSortKeysBuffer() const { return m_sortKeys.get(); }
    RWStructuredBuffer<TreeBuildDispatchArgsGpu>* GetDispatchArgsBuffer() const { return m_dispatchArgs.get(); }
    RWStructuredBuffer<DispatchArgsGpu>* GetIndirectDispatchArgsBuffer() const { return m_indirectDispatchArgs.get(); }
    RWStructuredBuffer<uint32_t>* GetNodeVisitedBuffer() const { return m_nodeVisited.get(); }


private:
    // Shared heap + tree root signature + every root resource. False = cannot run.
    bool BindRoots();

    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();
    void CreateCommandSignature();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass>             m_buildPass;
    std::shared_ptr<VxpgClusterPass>                   m_clusterPass;
    std::shared_ptr<VxpgClusterVisibilityPass>         m_clusterVisibilityPass;

    BitonicSortPass m_sort;

    std::unique_ptr<RWStructuredBuffer<uint64_t>>                 m_sortKeys;      // SIByL u_Codes
    std::unique_ptr<RWStructuredBuffer<LightTreeNodeGpu>>         m_nodes;         // SIByL u_Nodes
    std::unique_ptr<RWStructuredBuffer<uint32_t>>                 m_leafRanges;    // SIByL u_Descendant (dead)
    std::unique_ptr<RWStructuredBuffer<int32_t>>                  m_compactToLeaf; // SIByL compact2leaf
    std::unique_ptr<RWStructuredBuffer<int32_t>>                  m_clusterRoots;  // SIByL cluster_roots
    std::unique_ptr<RWStructuredBuffer<TreeBuildDispatchArgsGpu>> m_dispatchArgs;  // SIByL u_ConstrIndirectArgs
    std::unique_ptr<RWStructuredBuffer<DispatchArgsGpu>>          m_indirectDispatchArgs; // Bamboo: live-sized group counts
    std::unique_ptr<RWStructuredBuffer<uint32_t>>                 m_nodeVisited;   // merge sibling-gate (own scalar buffer)
    std::unique_ptr<RWStructuredBuffer<float>>                   m_spixelClusterHeap; // SIByL tltree (mapX*mapY*64)

    RootSignature m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_dispatchCommandSignature;
    ComputeProgram* m_clearProgram = nullptr;
    ComputeProgram* m_encodeProgram = nullptr;
    ComputeProgram* m_initialProgram = nullptr;
    ComputeProgram* m_internalProgram = nullptr;
    ComputeProgram* m_mergeProgram = nullptr;
    ComputeProgram* m_topLevelProgram = nullptr;

    uint32_t m_mapX = 0;
    uint32_t m_mapY = 0;
    bool m_initialized = false;
};
