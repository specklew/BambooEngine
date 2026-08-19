#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"

// Reusable GPU bitonic sort over uint64 keys (MiniEngine-style, ported from
// SIByL bitonicsort/). Sorts up to 65536 keys ascending. The valid element
// count is read live from a caller-supplied counter buffer inside each kernel, and
// the ladder is issued through ExecuteIndirect off per-stage group counts the
// caller's producer wrote: the live count pads up to a power of two and stages
// above it get zero groups. A frame lighting 1500 voxels runs ONE presort group
// where this used to dispatch 21 stages x 32 groups unconditionally (ADR 0003
// option b's in-shader early-outs stay, as the guard for the tail inside a group).
class BitonicSortPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Sorts keyBuffer in place. numElements is read from counterBuffer at
    // counterByteOffset (the light-tree args' clamped valid-count field). The
    // caller must rebind its own root signature afterwards.
    // stageArgsBuffer holds BITONIC_SORT_STAGE_COUNT DISPATCH argument triples in
    // ladder order starting at stageArgsByteOffset, and must already be in
    // INDIRECT_ARGUMENT state.
    void Sort(ID3D12Resource*           keyBuffer,
              D3D12_GPU_VIRTUAL_ADDRESS  keyBufferVA,
              D3D12_GPU_VIRTUAL_ADDRESS  counterBufferVA,
              uint32_t                   counterByteOffset,
              ID3D12CommandSignature*    dispatchSignature,
              ID3D12Resource*            stageArgsBuffer,
              uint64_t                   stageArgsByteOffset);

    // Elements the 65536-network sorts (also the sort-key buffer capacity).
    static constexpr uint32_t kCapacity = 65536;

private:
    void CreateRootSignature();
    void CreatePSOs();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;

    RootSignature m_rootSig;
    ComputeProgram* m_presortProgram = nullptr;
    ComputeProgram* m_outerProgram = nullptr;
    ComputeProgram* m_innerProgram = nullptr;

    bool m_initialized = false;
};
