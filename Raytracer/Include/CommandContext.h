#pragma once
#include <vector>

#include "ShaderProgram.h"

class Resource;

// Stands in for the command list a pass used to cache at Initialize. Which list a
// pass records into stopped being knowable up front in ADR 0017 phase 6c: the
// frame is a run per queue, each on its own list, so the answer is "whichever one
// the render graph has bound for the run this node belongs to". Assigning a list
// to it is deliberately a no-op — the parameter is kept only so a pass's
// Initialize signature still reads the same on both sides of the change.
//
// Every accessor routes through CommandContext::GetCommandList(), which also
// flushes pending barriers, so a pass can never record ahead of its own barriers.
class ActiveCommandList
{
public:
    ActiveCommandList& operator=(const Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4>&) { return *this; }

    [[nodiscard]] ID3D12GraphicsCommandList4* Get() const;
    ID3D12GraphicsCommandList4* operator->() const { return Get(); }
    operator ID3D12GraphicsCommandList4*() const { return Get(); }
};

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

    // For recording that issues no GPU work and therefore needs no barrier flush
    // (PIX markers). Keeps a queued batch intact so it can still merge with the
    // work that follows.
    [[nodiscard]] ID3D12GraphicsCommandList4* GetCommandListUnflushed() const { return m_commandList; }

    // Barriers — tracked resources go through the phase-0 checked path.
    void Transition(Resource& resource, D3D12_RESOURCE_STATES expectedBefore, D3D12_RESOURCE_STATES after);
    void UavBarrier(Resource& resource);
    // Raw overloads for resources that are still bare ComPtrs (VXPG textures);
    // they carry no tracked state, so nothing is checked.
    void TransitionRaw(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    void UavBarrierRaw(ID3D12Resource* resource);

    // Already-built barrier from the render graph's compiled plan. The graph
    // updated the tracker when it built it, so nothing is checked again here.
    void EnqueueBarrier(const D3D12_RESOURCE_BARRIER& barrier, ID3D12Resource* resource);

    void FlushBarriers();
    [[nodiscard]] bool HasPendingBarriers() const { return !m_pendingBarriers.empty(); }

    // Binding
    void SetComputeProgram(const ComputeProgram& program);

    // Work — every one of these flushes pending barriers first.
    void Dispatch(uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ);
    void DispatchIndirect(ID3D12CommandSignature* commandSignature, ID3D12Resource* argumentBuffer, uint64_t argumentBufferOffset = 0);
    void DispatchRays(const D3D12_DISPATCH_RAYS_DESC& desc);
    void DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndex, int32_t baseVertex, uint32_t startInstance);
    void CopyResource(ID3D12Resource* destination, ID3D12Resource* source);
    void CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION& destination, const D3D12_TEXTURE_COPY_LOCATION& source);

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
