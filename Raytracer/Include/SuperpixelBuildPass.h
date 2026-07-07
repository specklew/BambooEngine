#pragma once

// VXPG V2 Stage B: superpixel clustering (SLIC over the ShadingPoints G-buffer).
// Per frame: InitSeedCenters -> N x [FindCenterAssociation -> SumCenter] ->
// ClearCounter -> final FindCenterAssociation (gather). See docs/adr/0002.
// Outputs (private heap): per-pixel index, representative centers, per-superpixel
// pixel-lists. Index + center are also exposed to the raster debug views
// (modes 15/16) via the main heap.
class SuperpixelBuildPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Recreates resolution-dependent buffers and rebinds the private heap (slot 0 =
    // the injection ShadingPoints UAV). Call after LightInjectionPass::OnResize.
    void OnResize(uint32_t width, uint32_t height, ID3D12Resource* shadingPoints);

    // shadingPoints is re-pointed into the private heap if it changed since the last
    // call (injection recreates it on resize/scene change/shader reload).
    void Run(ID3D12Resource* shadingPoints, float weight, float posNormalizer);

    // Debug views read these from the main heap; Renderer writes the UAVs on resize.
    void WriteIndexUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;
    void WriteCenterUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;

    // Raw resources for the cluster-visibility pass (per-superpixel pixel lists +
    // the per-pixel superpixel id map), which creates its own descriptors.
    ID3D12Resource* GetGatheredResource() const { return m_gathered.Get(); }
    ID3D12Resource* GetCounterResource() const { return m_counter.Get(); }
    ID3D12Resource* GetIndexResource() const { return m_index.Get(); }
    uint32_t GetMapX() const { return m_mapX; }
    uint32_t GetMapY() const { return m_mapY; }

private:
    void CreateRootSignature();
    void CreatePSOs();
    void CreateBuffers();
    void CreatePrivateHeap(ID3D12Resource* shadingPoints);

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_initPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_assocPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_sumPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_privateHeap;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_center;       // map_size  RGBA32F
    Microsoft::WRL::ComPtr<ID3D12Resource> m_index;        // screen    R32_SINT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_counter;      // map_size  R32_UINT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gathered;     // map*size  RG32_SINT

    ID3D12Resource* m_boundShadingPoints = nullptr; // raw: lifetime owned by injection

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    uint32_t m_mapX   = 0;
    uint32_t m_mapY   = 0;
    bool     m_initialized = false;
};
