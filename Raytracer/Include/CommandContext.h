#pragma once
#include <vector>

#include "ShaderProgram.h"

class Resource;

// ADR 0017 L3: the only thing that submits work to the direct command list.
// Barriers are queued rather than emitted one at a time and flushed as a single
// ResourceBarrier call right before the work that needs them — which is also the
// shape the phase-3 graph will emit, so passes converted here need no second
// rewrite.
//
// One instance per command list; the engine records into a single direct list, so
// Get() is that list's context. Anything not yet converted goes through
// GetCommandList(), which flushes first so a raw call can never overtake a queued
// barrier.
class CommandContext
{
public:
    static CommandContext& Get();

    void Bind(ID3D12GraphicsCommandList4* commandList);

    // Escape hatch for unconverted recording (root bindings, viewport state, ...).
    // Flushes pending barriers so raw work cannot slip in ahead of them.
    ID3D12GraphicsCommandList4* GetCommandList();

    // Barriers — tracked resources go through the phase-0 checked path.
    void Transition(Resource& resource, D3D12_RESOURCE_STATES expectedBefore, D3D12_RESOURCE_STATES after);
    void UavBarrier(Resource& resource);
    // Raw overloads for resources that are still bare ComPtrs (VXPG textures);
    // they carry no tracked state, so nothing is checked.
    void TransitionRaw(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    void UavBarrierRaw(ID3D12Resource* resource);

    void FlushBarriers();
    [[nodiscard]] bool HasPendingBarriers() const { return !m_pendingBarriers.empty(); }

    // Binding
    void SetComputeProgram(const ComputeProgram& program);

    // Work — every one of these flushes pending barriers first.
    void Dispatch(uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ);
    void DispatchIndirect(ID3D12CommandSignature* commandSignature, ID3D12Resource* argumentBuffer,
                          uint64_t argumentBufferOffset = 0);
    void DispatchRays(const D3D12_DISPATCH_RAYS_DESC& desc);
    void DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndex,
                              int32_t baseVertex, uint32_t startInstance);
    void CopyResource(ID3D12Resource* destination, ID3D12Resource* source);

    void Close();

private:
    // Two barriers on one resource must not share a ResourceBarrier call —
    // barriers inside a call are unordered, so the second one queued forces the
    // batch out first and starts a new one.
    void QueueBarrier(const D3D12_RESOURCE_BARRIER& barrier, ID3D12Resource* resource);

    ID3D12GraphicsCommandList4*         m_commandList = nullptr;
    std::vector<D3D12_RESOURCE_BARRIER> m_pendingBarriers;
    std::vector<ID3D12Resource*>        m_pendingBarrierResources;
};
