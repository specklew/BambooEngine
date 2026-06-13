#pragma once

class Scene;

struct VoxelGridConstants
{
    DirectX::XMFLOAT3 gridMin;
    float             voxelSize;
    DirectX::XMFLOAT3 gridMax;
    uint32_t          gridDim;
    uint32_t          injectUseAvg;
    uint32_t          _reserved0; // kept for HLSL cbuffer layout compatibility
    float             heatScale;
    uint32_t          _pad0;
};

class VoxelizationPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        Microsoft::WRL::ComPtr<ID3D12RootSignature>        rasterRootSignature);

    void OnSceneLoaded(const Scene& scene);
    void RunFrame(const Scene& scene, uint32_t requestedGridDim);

    bool DidResize() const { return m_didResize; }

    // Runtime knobs propagated to the shared grid constant buffer each frame
    void SetRuntimeParams(bool injectUseAvg, float heatScale);

    Microsoft::WRL::ComPtr<ID3D12Resource> GetOccupancyTexture() const { return m_occupancyTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetIrradianceTexture() const { return m_irradianceTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetVplCountTexture() const { return m_vplCountTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetGridConstantsBuffer() const { return m_gridConstantsCB; }
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> GetDescriptorHeap() const { return m_descHeap; } // [0]=occupancy [1]=irradiance [2]=vpl count
    uint32_t GetGridDim() const { return m_gridDim; }
    const VoxelGridConstants&              GetGridConstants() const { return m_gridConstants; }

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
    void DispatchClear();
    void DispatchVoxelize(const Scene& scene);
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

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_clearRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_voxelizeRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_voxelizePso;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap; // [0]=occupancy UAV [1]=irradiance UAV [2]=vpl count UAV

    uint32_t m_gridDim     = 64;
    bool     m_initialized = false;
    bool     m_haveScene   = false;
    bool     m_didResize   = false;

    DirectX::XMFLOAT3 m_cachedAabbMin{};
    DirectX::XMFLOAT3 m_cachedAabbMax{};
};
