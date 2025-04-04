#pragma once
#include "Constants.h"

class Renderer
{
public:
	Renderer() = default;
	
	void Initialize();
	void Update(double elapsedTime, double totalTime);
	void Render(double elapsedTime, double totalTime);
	void CleanUp();
	void OnMouseMove(unsigned long long btnState, int x, int y);

private:

	
	void SetupDeviceAndDebug();
	void CreateCommandQueue();
	void CreateCommandAllocators();
	void CreateFence();
	void CreateSwapChain();
	void CreateCommandList();

	void ResetCommandList();

	void CreateRTVDescriptorHeap();
	void CreateRenderTargetViews();

	void CreateDepthStencilView();
	void CreateDSVDescriptorHeap();

	void CreateVertexAndIndexBuffer();

	void CreateConstantBufferView();

	void CreateRootSignature();

	void CreatePipelineState();

	void LoadShaders();

	void SetViewport();

	void FlushCommandQueue();
	
	bool CheckTearingSupport();
	
	Microsoft::WRL::ComPtr<IDXGIAdapter4> GetHardwareAdapter(bool useWarp = false);
	Microsoft::WRL::ComPtr<ID3D12Device2> GetDeviceForAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter);

	Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(const void* initData, UINT64 byteSize, Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer);

	Microsoft::WRL::ComPtr<ID3D12Debug1> m_spDebugController;
	
	Microsoft::WRL::ComPtr<IDXGIAdapter4> m_dxgiAdapter;
	Microsoft::WRL::ComPtr<ID3D12Device2> m_d3d12Device;
	Microsoft::WRL::ComPtr<IDXGIFactory6> m_dxgiFactory;
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_d3d12CommandQueue;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_d3d12CommandAllocators[NUM_FRAMES];
	Microsoft::WRL::ComPtr<ID3D12Fence> m_d3d12Fence;
	Microsoft::WRL::ComPtr<IDXGISwapChain3> m_dxgiSwapChain;
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_d3d12CommandList;

	DXGI_FORMAT m_backBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT m_depthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
	
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_d3d12RTVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_d3d12RenderTargets[NUM_FRAMES];
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_d3d12DSVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_depthStencilBuffer;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBuffer;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferUploader;
	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
	D3D12_VERTEX_BUFFER_VIEW m_vertexBuffers[1];

	Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBuffer;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferUploader;
	D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_constantBuffer;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_CBVDescriptorHeap;
	BYTE* m_mappedData = nullptr;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;

	Microsoft::WRL::ComPtr<ID3DBlob> m_pixelShader;
	Microsoft::WRL::ComPtr<ID3DBlob> m_vertexShader;

	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineStateObject;

	UINT m_frameIndex = 0;
	UINT64 m_fenceValues[NUM_FRAMES] = {};
	HANDLE m_fenceEvent = nullptr;

	UINT m_rtvDescriptorSize = 0;

	int m_lastMousePosX = 0;
	int m_lastMousePosY = 0;
	float m_theta = 0;
	float m_phi = 0;
	float m_radius = 10;

	DirectX::XMFLOAT4X4 m_view = {};
	DirectX::XMFLOAT4X4 m_world = {};
	DirectX::XMFLOAT4X4 m_proj = {};
};
