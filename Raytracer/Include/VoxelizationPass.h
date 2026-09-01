#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"

#include "Resources/RWStructuredBuffer.h"

class Scene;
class GpuMemoryReport;

struct VoxelGridConstants
{
    DirectX::XMFLOAT3 gridMin;
    float             voxelSize;
    DirectX::XMFLOAT3 gridMax;
    uint32_t          gridDim;
    uint32_t          injectUseAvg;
    // Trace the injection bounce for every Nth pixel in each axis (1 = every pixel).
    uint32_t          injectPixelStride;
    float             heatScale;
};

// Geometry bake + per-frame injection-accumulator clear (ADR 0004). The scene
// is conservative-rasterized into occupancy + quantized per-voxel bounds ONCE
// per bake; a bake is invalidated by scene load, grid resize, or a bound-flag
// change. Per frame only the irradiance/VPL-count clear runs.
class VoxelizationPass
{
public:
    // P5: the resources this stage holds, for the guiding chain's memory report.
    // Declared per pass rather than collected from a base class because ADR 0017 left
    // most of this chain as raw ComPtr, which no base class ever sees.
    void ReportMemory(GpuMemoryReport& report) const;
    void Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    void OnSceneLoaded(const Scene& scene);

    // Resizes and invalidates, recording nothing: a grid resize recreates the
    // textures the graph is about to import, so it has to settle before the
    // imports. Returns whether the bake nodes below still owe this frame a bake.
    bool PrepareFrame(const Scene& scene, uint32_t requestedGridDim, bool bakeUseCompact, bool bakeClipping);

    bool DidResize() const { return m_didResize; }

    // Runtime knobs propagated to the shared grid constant buffer each frame
    void SetRuntimeParams(bool injectUseAvg, float heatScale, uint32_t injectPixelStride);

    // Zeroes the per-frame injection accumulators (irradiance + VPL count).
    // Ordered by the graph, always ahead of the injection trace that refills them:
    // no VPL data survives a frame boundary.
    void DispatchFrameClear();

    // The two halves of the geometry bake, one graph node each. The bake stays
    // valid until a scene load, grid resize or bound-flag change invalidates it,
    // and marks itself valid only once it has actually run — a frame that culls it
    // simply owes the bake again.
    void DispatchBakeClear();
    void DispatchBake(const Scene& scene);

    Microsoft::WRL::ComPtr<ID3D12Resource> GetOccupancyTexture() const { return m_occupancyTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetIrradianceTexture() const { return m_irradianceTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplCountTexture() const { return m_vplCountTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetGridConstantsBuffer() const { return m_gridConstantsCB; }
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
    void WriteGridConstantsCB();
    void WriteUintTex3DUav(ID3D12Resource* resource, D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void RecreateForNewDim(uint32_t newDim);

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_occupancyTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_irradianceTex; // packed fixed-point irradiance (x100), uint atomics
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vplCountTex;   // VPL count per voxel for averaging
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gridConstantsCB;
    VoxelGridConstants                     m_gridConstants{};
    void*                                  m_gridConstantsCBMapped = nullptr;

    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_bakedBoundMin;
    std::unique_ptr<RWStructuredBuffer<uint32_t>> m_bakedBoundMax;

    RootSignature m_clearRootSig;     // per-frame accumulator clear
    ComputeProgram* m_clearProgram = nullptr;
    RootSignature m_bakeClearRootSig;
    ComputeProgram* m_bakeClearProgram = nullptr;
    RootSignature m_bakeRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_bakePso;

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
