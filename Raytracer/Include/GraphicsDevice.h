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

    void CreateCommandQueue();
    void CreateCommandAllocators();
    void CreateFence();
    void CreateSwapChain(HWND windowHandle, uint32_t width, uint32_t height, DXGI_FORMAT backBufferFormat);

    bool CheckTearingSupport();
    bool CheckRayTracingSupport() const;

    // Signal + CPU-wait until the queue drains, then refresh the frame index.
    void FlushCommandQueue();
    void RefreshFrameIndex();

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Device5>& GetDevice() { return m_device; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandQueue>& GetCommandQueue() { return m_commandQueue; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& GetCommandAllocator(UINT frameIndex) { return m_commandAllocators[frameIndex]; }
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
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocators[Constants::Graphics::NUM_FRAMES];

    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;

    Microsoft::WRL::ComPtr<IDXGISwapChain3> m_swapChain;
    UINT m_frameIndex = 0;
    bool m_tearingSupport = false;
};
