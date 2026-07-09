#pragma once

#include "Resources/RWStructuredBuffer.h"

class VoxelGuidingBuildPass;
class LightInjectionPass;

// VXPG fingerprint pass (MRCS column reduction): assigns every compacted lit
// voxel a 128-bit visibility signature. Two compute kernels run each frame
// after the guiding distribution is built:
//   SampleScreenRepresentatives -> picks 128 stratified screen points + emits
//                                   the downstream guiding dispatch args
//   BuildVoxelFingerprints      -> one shadow ray per (representative, voxel);
//                                   the visibility bits ARE the fingerprint
// The visibility kernel uses inline RayQuery in compute (DXR 1.1) and
// [WaveSize(32)] ballot packing, so it requires SM 6.6 + RT Tier 1.1.
// The visibility kernel dispatches indirectly off gGuidingDispatchArgs[2] =
// (4, ceil(litVoxelCount/8), 1), sized to the live lit-voxel count rather than
// the worst-case grid capacity (ADR 0003 option b retrofit).
class VxpgFingerprintPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<VoxelGuidingBuildPass>             buildPass,
        std::shared_ptr<LightInjectionPass>                injectionPass);

    // Resolution of the ShadingPoints G-buffer (drives stratification).
    void OnResize(uint32_t width, uint32_t height);

    // tlasVa = the scene TLAS result buffer for the inline visibility rays.
    void Run(D3D12_GPU_VIRTUAL_ADDRESS tlasVa, uint32_t frameIndex);

    RWStructuredBuffer<DirectX::XMFLOAT4>* GetScreenRepresentativePointsBuffer() const { return m_screenRepresentativePoints.get(); }
    RWStructuredBuffer<uint32_t>*          GetVoxelFingerprintsBuffer() const { return m_voxelFingerprints.get(); }
    RWStructuredBuffer<DirectX::XMUINT4>*  GetGuidingDispatchArgsBuffer() const { return m_guidingDispatchArgs.get(); }

private:
    void CreateBuffers();
    void CreatePrivateHeap();
    void RebindShadingPointsIfChanged();
    void CreateRootSignatures();
    void CreatePSOs();
    void CreateCommandSignature();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    std::shared_ptr<VoxelGuidingBuildPass>             m_buildPass;
    std::shared_ptr<LightInjectionPass>                m_injectionPass;

    std::unique_ptr<RWStructuredBuffer<DirectX::XMFLOAT4>> m_screenRepresentativePoints;
    std::unique_ptr<RWStructuredBuffer<DirectX::XMUINT4>>  m_guidingDispatchArgs;
    std::unique_ptr<RWStructuredBuffer<uint32_t>>          m_voxelFingerprints;

    // Private heap slot 0 = ShadingPoints UAV (presample reads it).
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap;
    ID3D12Resource* m_boundShadingPoints = nullptr;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_presampleRootSig;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_visibilityRootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_presamplePso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_visibilityPso;

    // Dispatch-indirect signature for the visibility kernel (pure DISPATCH arg,
    // no root-argument changes, so pRootSignature is null and it is reusable).
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_dispatchCommandSignature;

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    bool     m_initialized = false;
};
