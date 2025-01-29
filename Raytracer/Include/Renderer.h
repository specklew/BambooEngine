#pragma once
#include "Constants.h"

class Renderer
{
public:
	void Initialize();
	void Update(double elapsedTime, double totalTime);
	void Render(double elapsedTime, double totalTime);

private:
	void SetupDevice();
	void CreateCommandQueue();
	void CreateCommandAllocators();
	void CreateFence();

	bool CheckTearingSupport();
	
	Microsoft::WRL::ComPtr<IDXGIAdapter4> GetHardwareAdapter(bool useWarp = false);
	Microsoft::WRL::ComPtr<ID3D12Device2> GetDeviceForAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter);
	
	Microsoft::WRL::ComPtr<IDXGIAdapter4> m_dxgiAdapter;
	Microsoft::WRL::ComPtr<ID3D12Device2> m_d3d12Device;
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_d3d12CommandQueue;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_d3d12CommandAllocators[NUM_FRAMES];
	Microsoft::WRL::ComPtr<ID3D12Fence> m_d3d12Fence;

	UINT m_frameIndex = 0;
	UINT64 m_fenceValues[NUM_FRAMES] = {};
	HANDLE m_fenceEvent = nullptr;
};
