#pragma once

#include "Resources/RWStructuredBuffer.h"

class VoxelizationPass;

// VXPG guiding distribution build (compute): reloads baked per-voxel bounds
// for lit voxels and compacts nonzero-irradiance voxels into a flat list (with
// compact-indexed representative VPLs and area-premultiplied irradiance).
// Runs each frame after light injection; consumed by GuidedPathTracingPass
// and the passes downstream. (V1's flat-CDF kernel retired — voxel selection
// lives in the light tree now.)
class VoxelGuidingBuildPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelizationPass>                  voxelPass);

    // representativeTex = injection's per-voxel representative VPL Texture3D;
    // re-bound into the private heap when it changes (recreated on resize).
    void Run(ID3D12Resource* representativeTex);

    // Recreates the grid-sized buffers after a voxel-grid resize. Caller must
    // have flushed the GPU first (the old buffers may be in flight).
    void OnVoxelGridResize();

    // [0] = compacted voxel count ([1] retired with the flat CDF)
    RWStructuredBuffer<uint32_t>* GetCountersBuffer() const { return m_counters.get(); }
    RWStructuredBuffer<uint32_t>* GetCompactIdsBuffer() const { return m_compactIds.get(); }
    RWStructuredBuffer<int32_t>*  GetInverseIndexBuffer() const { return m_inverseIndex.get(); }
    RWStructuredBuffer<DirectX::XMFLOAT4>* GetCompactVoxelLightPointsBuffer() const { return m_compactVoxelLightPoints.get(); }
    RWStructuredBuffer<float>*    GetPremulIrradianceBuffer() const { return m_premulIrradiance.get(); }
    RWStructuredBuffer<DirectX::XMUINT4>* GetLiveBoundMinBuffer() const { return m_liveBoundMin.get(); }
    RWStructuredBuffer<DirectX::XMUINT4>* GetLiveBoundMaxBuffer() const { return m_liveBoundMax.get(); }

private:
    void CreateBuffers();
    void CreateGridSizedBuffers();
    void CreateDescriptorHeap();
    void RebindDescriptorsIfChanged(ID3D12Resource* representativeTex);
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;

    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_counters;
    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_compactIds;
    std::unique_ptr<RWStructuredBuffer<int32_t>>  m_inverseIndex;      // grid-sized
    std::unique_ptr<RWStructuredBuffer<DirectX::XMUINT4>> m_liveBoundMin; // grid-sized
    std::unique_ptr<RWStructuredBuffer<DirectX::XMUINT4>> m_liveBoundMax; // grid-sized
    std::unique_ptr<RWStructuredBuffer<DirectX::XMFLOAT4>> m_compactVoxelLightPoints; // SIByL u_RepresentVPL
    std::unique_ptr<RWStructuredBuffer<float>>    m_premulIrradiance;

    // Private heap: [0]=irradiance [1]=vpl count [2]=representative VPL tex.
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap;
    ID3D12Resource* m_boundIrradiance     = nullptr; // raw: change detection only
    ID3D12Resource* m_boundVplCount       = nullptr;
    ID3D12Resource* m_boundRepresentative = nullptr;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_reloadPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_compactPso;

    bool m_initialized = false;
};
