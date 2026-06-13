#pragma once

#include "Resources/RWStructuredBuffer.h"

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
    RWStructuredBuffer<uint32_t>* GetCountersBuffer() const { return m_counters.get(); }
    RWStructuredBuffer<uint32_t>* GetCompactIdsBuffer() const { return m_compactIds.get(); }
    RWStructuredBuffer<float>*    GetCdfBuffer() const { return m_cdf.get(); }

private:
    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;

    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_counters;
    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_compactIds;
    std::unique_ptr<RWStructuredBuffer<float>>    m_weights;
    std::unique_ptr<RWStructuredBuffer<float>>    m_cdf;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_compactPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_cdfPso;

    bool m_initialized = false;
};
