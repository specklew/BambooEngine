#pragma once

// Reusable GPU bitonic sort over uint64 keys (MiniEngine-style, ported from
// SIByL bitonicsort/). Sorts up to 65536 keys ascending. The valid element
// count is read live from a caller-supplied counter buffer each dispatch, so
// the network runs as fixed worst-case dispatches (21 stages x 32 groups) with
// in-shader early-outs (ADR 0003 option b) — no ExecuteIndirect needed.
class BitonicSortPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Sorts keyBuffer in place. numElements is read from counterBuffer at
    // counterByteOffset (the light-tree args' clamped valid-count field). The
    // caller must rebind its own root signature afterwards.
    void Sort(ID3D12Resource*           keyBuffer,
              D3D12_GPU_VIRTUAL_ADDRESS  keyBufferVA,
              D3D12_GPU_VIRTUAL_ADDRESS  counterBufferVA,
              uint32_t                   counterByteOffset);

    // Elements the 65536-network sorts (also the sort-key buffer capacity).
    static constexpr uint32_t kCapacity = 65536;

private:
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_presortPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_outerPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_innerPso;

    bool m_initialized = false;
};
