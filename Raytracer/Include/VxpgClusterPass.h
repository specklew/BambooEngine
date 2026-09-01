#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"

#include "Resources/RWStructuredBuffer.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgFingerprintPass;
class GpuMemoryReport;

// VXPG cluster pass (MRCS column clustering): groups the fingerprinted lit
// voxels into 32 supervoxels. Two compute kernels run each frame after the
// fingerprint pass:
//   SeedClusterCenters  -> k-means++ seeding, one 1024-thread group picking 32
//                          seeds spread apart in fingerprint+intensity space
//   AssignVoxelClusters -> every compact voxel stores its nearest cluster id
//   Accumulate/Update   -> the Lloyd half: sum each cluster's members, replace
//                          the center with their centroid, assign again
// SIByL stops after seeding, arguing a bitmask has no mean. Under Hamming it
// does — the per-bit majority — so the iteration is well defined; see the header
// comment of vxpgCluster.hlsl. kLloydIterations rounds run after the seeding.
// The seeding kernel uses two-level wave reductions, so it requires SM 6.6.
class VxpgClusterPass
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
        std::shared_ptr<VxpgFingerprintPass>               fingerprintPass);

    // Lloyd rounds after the seeding; mirrors CLUSTER_LLOYD_ITERATIONS in the
    // shader. The renderer emits one node trio per round.
    static constexpr uint32_t kLloydIterations = 4;

    // One graph node per kernel; the dispatch-args state flip is declared.
    void RunSeed(uint32_t frameIndex);
    // Only the final round records diagnostics: the stats buffer is zeroed once,
    // by the seeding, so every assignment that wrote to it would be summed in.
    void RunAssign(uint32_t frameIndex, bool collectStats);
    void RunAccumulate(uint32_t frameIndex);
    void RunUpdate(uint32_t frameIndex);

    // One-shot cluster diagnostics (vxpg.cluster.dumpStats). Armed, the assign
    // kernel counts each cluster's members and sums the two distance terms it
    // actually realised; the frame then reads them back and logs. Answers the
    // question a picture cannot: whether the grouping is driven by the visibility
    // fingerprint or by the irradiance term.
    [[nodiscard]] static bool IsStatsDumpArmed();
    void RecordStatsCopy();
    void ResolveStats();
    RWStructuredBuffer<uint32_t>* GetClusterStatsBuffer() const { return m_clusterStats.get(); }

    // SIByL svoxel_info: the descriptor each voxel is compared against.
    struct ClusterCenter
    {
        DirectX::XMUINT4  fingerprint;
        DirectX::XMFLOAT3 position;
        float             intensity;
    };

    RWStructuredBuffer<int32_t>*       GetClusterSeedCompactIdsBuffer() const { return m_clusterSeedCompactIds.get(); }
    RWStructuredBuffer<ClusterCenter>* GetClusterCentersBuffer() const { return m_clusterCenters.get(); }
    RWStructuredBuffer<int32_t>*       GetVoxelClusterAssignmentsBuffer() const { return m_voxelClusterAssignments.get(); }
    RWStructuredBuffer<uint32_t>*      GetClusterAccumulatorsBuffer() const { return m_clusterAccumulators.get(); }


private:
    // Both kernels share one root signature but separate nodes may have barriers
    // between them, so each re-binds. False = cannot run.
    bool BindCommon(uint32_t frameIndex, bool collectStats);

    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();
    void CreateCommandSignature();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass>             m_buildPass;
    std::shared_ptr<VxpgFingerprintPass>               m_fingerprintPass;

    std::unique_ptr<RWStructuredBuffer<int32_t>>       m_clusterSeedCompactIds;   // SIByL u_Seeds
    std::unique_ptr<RWStructuredBuffer<ClusterCenter>> m_clusterCenters;          // SIByL u_RowClusterInfo
    std::unique_ptr<RWStructuredBuffer<int32_t>>       m_voxelClusterAssignments; // SIByL u_Clusters
    std::unique_ptr<RWStructuredBuffer<uint32_t>>      m_clusterStats;            // diagnostic, no SIByL counterpart
    std::unique_ptr<RWStructuredBuffer<uint32_t>>      m_clusterAccumulators;     // Lloyd sums, no SIByL counterpart
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_clusterStatsReadback;
    uint32_t                                           m_statsRetries = 0;

    RootSignature m_rootSig;
    ComputeProgram* m_seedProgram = nullptr;
    ComputeProgram* m_assignProgram = nullptr;
    ComputeProgram* m_accumulateProgram = nullptr;
    ComputeProgram* m_updateProgram = nullptr;

    // Dispatch-indirect signature for the assign kernel (pure DISPATCH arg).
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_dispatchCommandSignature;

    bool m_initialized = false;
};
