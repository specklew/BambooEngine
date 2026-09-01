#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"
class GpuMemoryReport;

// VXPG V2 Stage B: superpixel clustering (SLIC over the ShadingPoints G-buffer).
// Per frame: InitSeedCenters -> N x [FindCenterAssociation -> SumCenter] ->
// ClearCounter -> final FindCenterAssociation (gather). See docs/adr/0002.
// Outputs: per-pixel index, representative centers, per-superpixel pixel-lists,
// all in the global heap — index and center are read from there by the debug
// views too, so there is one descriptor per resource rather than two.
class SuperpixelBuildPass
{
public:
    // P5: the resources this stage holds, for the guiding chain's memory report.
    // Declared per pass rather than collected from a base class because ADR 0017 left
    // most of this chain as raw ComPtr, which no base class ever sees.
    void ReportMemory(GpuMemoryReport& report) const;
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Recreates resolution-dependent buffers. Call after LightInjectionPass::OnResize
    // (shadingPoints only gates the call — the pass reads it from the global heap),
    // and write the new resources to their global slots afterwards.
    void OnResize(uint32_t width, uint32_t height, ID3D12Resource* shadingPoints);

    // Frame inputs, set once before the nodes are added.
    void SetFrameInputs(float weight, float posNormalizer);

    // One graph node per SLIC kernel: init -> N x (associate, sum) -> clear
    // counter -> final associate with gather.
    void RunInitSeeds();
    void RunAssociate(bool writeGather);
    void RunSumCenters();
    void RunClearCounter();

    // Debug views read these from the main heap; Renderer writes the UAVs on resize.
    void WriteIndexUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void WriteCenterUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    // Fuzzy 4-nearest blend outputs for the guided integrator's mixture pdf.
    void WriteFuzzyWeightUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void WriteFuzzyIndexUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;

    // Raw resources for the cluster-visibility pass (per-superpixel pixel lists +
    // the per-pixel superpixel id map), which creates its own descriptors.
    ID3D12Resource* GetGatheredResource() const { return m_gathered.Get(); }
    ID3D12Resource* GetCounterResource() const { return m_counter.Get(); }
    ID3D12Resource* GetIndexResource() const { return m_index.Get(); }
    ID3D12Resource* GetCenterResource() const { return m_center.Get(); }
    ID3D12Resource* GetFuzzyWeightResource() const { return m_fuzzyWeight.Get(); }
    ID3D12Resource* GetFuzzyIndexResource() const { return m_fuzzyIndex.Get(); }
    uint32_t GetMapX() const { return m_mapX; }
    uint32_t GetMapY() const { return m_mapY; }


private:
    // Binds heap + root signature + constants; false = cannot run.
    bool BindCommon(bool writeGather);

    void CreateRootSignature();
    void CreatePSOs();
    void CreateBuffers();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;

    RootSignature m_rootSig;
    ComputeProgram* m_initProgram = nullptr;
    ComputeProgram* m_assocProgram = nullptr;
    ComputeProgram* m_sumProgram = nullptr;
    ComputeProgram* m_clearProgram = nullptr;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_center;       // map_size  RGBA32F
    Microsoft::WRL::ComPtr<ID3D12Resource> m_index;        // screen    R32_SINT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_counter;      // map_size  R32_UINT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gathered;     // map*size  RG32_SINT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fuzzyWeight;  // screen    RGBA32F
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fuzzyIndex;   // screen    RGBA32_SINT

    float m_weight        = 0.0f;
    float m_posNormalizer = 0.0f;

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    uint32_t m_mapX   = 0;
    uint32_t m_mapY   = 0;
    bool     m_initialized   = false;
    bool     m_buffersCreated = false;
};
