#pragma once
#include "Constants.h"

// L0 of the ADR 0017 layering: device, direct queue, command allocators, fence,
// swap chain, and capability checks. Owns nothing above the queue boundary —
// command lists, descriptor heaps and resources belong to the layers above.
class GraphicsDevice
{
public:
    // Debug layer + DRED + factory + adapter selection + device creation.
    void Initialize(bool enableDebugLayer);

    // Both queues: the direct one the frame is built on, and an async compute one
    // the render graph schedules independent chains onto (ADR 0017 phase 6c).
    void CreateCommandQueue();
    void CreateCommandAllocators();
    void CreateFence();
    void CreateSwapChain(HWND windowHandle, uint32_t width, uint32_t height, DXGI_FORMAT backBufferFormat);

    bool CheckTearingSupport();
    bool CheckRayTracingSupport() const;

    // Signal + CPU-wait until the queue drains, then refresh the frame index. For
    // structural changes only (resize, scene switch, shader reload, a readback):
    // the steady-state frame paces itself with the pair below instead.
    void FlushCommandQueue();
    void RefreshFrameIndex();

    // Frame pacing (ADR 0017 phase 6a). SignalFrame() marks the queue position of
    // the frame just submitted; WaitForCurrentFrame() blocks until whatever the
    // frame slot we are about to reuse last submitted has finished, which frees
    // its command allocator and its copy of the per-frame constants. That is
    // NUM_FRAMES-1 frames of run-ahead, where the flush allowed none.
    void SignalFrame();
    void WaitForCurrentFrame();

    // Cross-queue ordering. One monotonic fence both queues signal and wait on:
    // the value returned by a signal is what the other queue waits for, and the
    // wait is a queue operation, so it must sit between two submissions rather
    // than inside a command list.
    UINT64 SignalCrossQueue(ID3D12CommandQueue* queue);
    void   WaitCrossQueue(ID3D12CommandQueue* queue, UINT64 value);

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Device5>& GetDevice() { return m_device; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandQueue>& GetCommandQueue() { return m_commandQueue; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandQueue>& GetComputeQueue() { return m_computeQueue; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& GetCommandAllocator(UINT frameIndex) { return m_commandAllocators[frameIndex]; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& GetComputeCommandAllocator(UINT frameIndex) { return m_computeAllocators[frameIndex]; }
    [[nodiscard]] Microsoft::WRL::ComPtr<IDXGISwapChain3>& GetSwapChain() { return m_swapChain; }
    [[nodiscard]] UINT GetFrameIndex() const { return m_frameIndex; }
    [[nodiscard]] bool IsTearingSupported() const { return m_tearingSupport; }

private:
    Microsoft::WRL::ComPtr<IDXGIAdapter4> GetHardwareAdapter(bool useWarp = false);
    Microsoft::WRL::ComPtr<ID3D12Device5> GetDeviceForAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter);

    bool m_enableDebugLayer = true;

    Microsoft::WRL::ComPtr<IDXGIFactory6> m_dxgiFactory;
    Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> m_infoQueue;
    DWORD m_debugMessageCallbackCookie = 0;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_computeQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocators[Constants::Graphics::NUM_FRAMES];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_computeAllocators[Constants::Graphics::NUM_FRAMES];

    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue = 0;
    // Separate from m_fence so a cross-queue handshake never perturbs the frame
    // pacing values, which index by frame slot.
    Microsoft::WRL::ComPtr<ID3D12Fence> m_crossQueueFence;
    UINT64 m_crossQueueValue = 0;
    // Queue position of the last frame submitted from each slot; 0 means the slot
    // has never been used, so its first reuse waits for nothing.
    UINT64 m_frameFenceValues[Constants::Graphics::NUM_FRAMES] = {};
    HANDLE m_fenceEvent = nullptr;

    Microsoft::WRL::ComPtr<IDXGISwapChain3> m_swapChain;
    UINT m_frameIndex = 0;
    bool m_tearingSupport = false;
};
