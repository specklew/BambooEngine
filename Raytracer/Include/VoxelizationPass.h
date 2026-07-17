#pragma once

#include "Resources/RWStructuredBuffer.h"

class Scene;

struct VoxelGridConstants
{
    DirectX::XMFLOAT3 gridMin;
    float             voxelSize;
    DirectX::XMFLOAT3 gridMax;
    uint32_t          gridDim;
    uint32_t          injectUseAvg;
    uint32_t          supervoxelFactor; // SUPERVOXEL_GRID_FACTOR; supervoxel = voxelCoord / factor
    float             heatScale;
    uint32_t          reuseGiVpl; // 1 = VPL fitting samples come from last frame's guided-GI BSDF subtree (ADR 0009)
};

// Geometry bake + per-frame injection-accumulator clear (ADR 0004). The scene
// is conservative-rasterized into occupancy + quantized per-voxel bounds ONCE
// per bake; a bake is invalidated by scene load, grid resize, or a bound-flag
// change. Per frame only the irradiance/VPL-count clear runs.
class VoxelizationPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        Microsoft::WRL::ComPtr<ID3D12RootSignature>        rasterRootSignature);

    void OnSceneLoaded(const Scene& scene);

    // Rebakes if invalidated (grid resize / bound flags / scene load), then
    // clears the per-frame injection accumulators.
    void RunFrame(const Scene& scene, uint32_t requestedGridDim, bool bakeUseCompact, bool bakeClipping);

    bool DidResize() const { return m_didResize; }

    // Runtime knobs propagated to the shared grid constant buffer each frame
    void SetRuntimeParams(bool injectUseAvg, float heatScale, bool reuseGiVpl);

    // Zeroes the per-frame injection accumulators (irradiance + VPL count).
    // Called by the renderer: before injection in the faithful config, or
    // after the guiding build when VPL data is reused from last frame's GI
    // (ADR 0009) — the build passes must read before the wipe.
    void DispatchFrameClear();

    Microsoft::WRL::ComPtr<ID3D12Resource> GetOccupancyTexture() const { return m_occupancyTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetIrradianceTexture() const { return m_irradianceTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplCountTexture() const { return m_vplCountTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetGridConstantsBuffer() const { return m_gridConstantsCB; }
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> GetDescriptorHeap() const { return m_descHeap; } // [0]=occupancy [1]=irradiance [2]=vpl count
    uint32_t GetGridDim() const { return m_gridDim; }
    const VoxelGridConstants&              GetGridConstants() const { return m_gridConstants; }

    // Baked per-voxel bounds, 4 uints per cell, quantized to the voxel cube.
    RWStructuredBuffer<uint32_t>* GetBakedBoundMinBuffer() const { return m_bakedBoundMin.get(); }
    RWStructuredBuffer<uint32_t>* GetBakedBoundMaxBuffer() const { return m_bakedBoundMax.get(); }

    void WriteOccupancyUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void WriteIrradianceUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void WriteVplCountUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;

private:
    void CreateResources();
    void CreateRootSignatures();
    void CreatePSOs();
    void CreateDescriptorHeap();
    void WriteGridConstantsCB();
    void WriteUintTex3DUav(ID3D12Resource* resource, D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void DispatchBakeClear();
    void DispatchBake(const Scene& scene);
    void RecreateForNewDim(uint32_t newDim);

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>        m_rasterRootSignature; // existing raster root sig for color overlay binding

    Microsoft::WRL::ComPtr<ID3D12Resource> m_occupancyTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_irradianceTex; // packed fixed-point irradiance (x100), uint atomics
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vplCountTex;   // VPL count per voxel for averaging
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gridConstantsCB;
    VoxelGridConstants                     m_gridConstants{};
    void*                                  m_gridConstantsCBMapped = nullptr;

    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_bakedBoundMin;
    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_bakedBoundMax;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_clearRootSig;     // per-frame accumulator clear
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_bakeClearRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_bakeClearPso;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_bakeRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_bakePso;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap; // [0]=occupancy UAV [1]=irradiance UAV [2]=vpl count UAV

    uint32_t m_gridDim     = 64;
    bool     m_initialized = false;
    bool     m_haveScene   = false;
    bool     m_didResize   = false;
    bool     m_bakeValid   = false;
    bool     m_bakedUseCompact = false;
    bool     m_bakedClipping   = false;

    DirectX::XMFLOAT3 m_cachedAabbMin{};
    DirectX::XMFLOAT3 m_cachedAabbMax{};
};
