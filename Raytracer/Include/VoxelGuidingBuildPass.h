#pragma once

class VoxelizationPass;

// VXPG guiding distribution build (compute): compacts nonzero-irradiance
// voxels into a flat list and builds a prefix-sum CDF over their weights.
// Runs each frame after light injection; consumed by GuidedPathTracingPass.
class VoxelGuidingBuildPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelizationPass>                  voxelPass);

    void Run();

    // [0] = compacted voxel count, [1] = asuint(total weight)
    Microsoft::WRL::ComPtr<ID3D12Resource> GetCountersBuffer() const { return m_counters; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetCompactIdsBuffer() const { return m_compactIds; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetCdfBuffer() const { return m_cdf; }

private:
    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_counters;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_compactIds;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_weights;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cdf;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_compactPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_cdfPso;

    bool m_initialized = false;
};
