#pragma once

class VoxelizationPass;

// VXPG V2 Stage A: supervoxel clustering (compute).
// Supervoxels = coarse grid cells (voxelCoord / SUPERVOXEL_GRID_FACTOR). Each
// frame after the guiding build, accumulates per-supervoxel summed irradiance +
// active voxel count via atomics. Consumed by the guiding matrix build (Stage C).
class SupervoxelClusterPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelizationPass>                  voxelPass);

    void Run();

    Microsoft::WRL::ComPtr<ID3D12Resource> GetSupervoxelIrradianceBuffer() const { return m_svIrradiance; }
    Microsoft::WRL::ComPtr<ID3D12Resource> GetSupervoxelCountBuffer() const { return m_svCount; }

    // svDim = ceil(gridDim / SUPERVOXEL_GRID_FACTOR) for the current grid.
    uint32_t GetSupervoxelDim() const { return m_svDim; }

private:
    void CreateBuffers();
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelizationPass>                  m_voxelPass;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_svIrradiance;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_svCount;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_clearPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_accumPso;

    uint32_t m_svDim       = 0;
    bool     m_initialized = false;
};
