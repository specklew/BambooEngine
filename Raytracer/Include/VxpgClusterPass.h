#pragma once

#include "Resources/RWStructuredBuffer.h"

class VoxelizationPass;
class VoxelGuidingBuildPass;
class VxpgFingerprintPass;

// VXPG cluster pass (MRCS column clustering): groups the fingerprinted lit
// voxels into 32 supervoxels. Two compute kernels run each frame after the
// fingerprint pass:
//   SeedClusterCenters  -> k-means++ seeding, one 1024-thread group picking 32
//                          seeds spread apart in fingerprint+intensity space
//   AssignVoxelClusters -> every compact voxel stores its nearest cluster id
// Seeding-only k-means++ (no Lloyd iteration): bitmask centroids have no mean,
// so the seed descriptors ARE the cluster centers. The seeding kernel uses
// [WaveSize(32)] two-level wave reductions, so it requires SM 6.6.
class VxpgClusterPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelizationPass>                  voxelPass,
        std::shared_ptr<VoxelGuidingBuildPass>             buildPass,
        std::shared_ptr<VxpgFingerprintPass>               fingerprintPass);

    void Run(uint32_t frameIndex);

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

private:
    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;
    std::shared_ptr<VoxelGuidingBuildPass>             m_buildPass;
    std::shared_ptr<VxpgFingerprintPass>               m_fingerprintPass;

    std::unique_ptr<RWStructuredBuffer<int32_t>>       m_clusterSeedCompactIds;   // SIByL u_Seeds
    std::unique_ptr<RWStructuredBuffer<ClusterCenter>> m_clusterCenters;          // SIByL u_RowClusterInfo
    std::unique_ptr<RWStructuredBuffer<int32_t>>       m_voxelClusterAssignments; // SIByL u_Clusters

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_seedPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_assignPso;

    bool m_initialized = false;
};
