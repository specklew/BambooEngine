#pragma once

class Scene;

struct VoxelGridConstants
{
    DirectX::XMFLOAT3 gridMin;
    float             voxelSize;
    DirectX::XMFLOAT3 gridMax;
    uint32_t          gridDim;
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

    Microsoft::WRL::ComPtr<ID3D12Resource> GetOccupancyTexture() const { return m_occupancyTex; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetGridConstantsBuffer() const { return m_gridConstantsCB; }
    const VoxelGridConstants&              GetGridConstants() const { return m_gridConstants; }

    void WriteOccupancyUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const;

    static constexpr uint32_t kSh9Slices = 7;

private:
    void CreateResources();
    void CreateRootSignatures();
    void CreatePSOs();
    void CreateDescriptorHeap();
    void WriteGridConstantsCB();
    void DispatchClear();
    void DispatchVoxelize(const Scene& scene);

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>        m_rasterRootSignature; // existing raster root sig for color overlay binding

    Microsoft::WRL::ComPtr<ID3D12Resource> m_occupancyTex;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_sh9Textures[kSh9Slices];
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gridConstantsCB;
    VoxelGridConstants                     m_gridConstants{};
    void*                                  m_gridConstantsCBMapped = nullptr;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_clearRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_voxelizeRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_voxelizePso;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap; // [0]=occupancy UAV [1..7]=SH9 UAVs [8]=occupancy UAV-for-voxelize (alias)

    void RecreateForNewDim(uint32_t newDim);

    uint32_t m_gridDim     = 64;
    bool     m_initialized = false;
    bool     m_haveScene   = false;
    bool     m_didResize   = false;

    DirectX::XMFLOAT3 m_cachedAabbMin{};
    DirectX::XMFLOAT3 m_cachedAabbMax{};
};
