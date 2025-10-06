#pragma once
#include <iosfwd>
#include <vector>

#include "Constants.h"
#include "Helpers.h"
#include "InputElements.h"
struct Primitive;
class DescriptorHeapAllocator;
class RaytracePass;
class AccelerationStructures;

class Renderer
{
public:
	Renderer() = default;
	
	void Initialize();
	void Update(double elapsedTime, double totalTime);
	void Render(double elapsedTime, double totalTime);
	void CleanUp();
	void OnMouseMove(unsigned long long btnState, int x, int y);
	void ToggleRasterization();

	std::shared_ptr<Primitive> CreatePrimitive(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

private:
	void SetupDeviceAndDebug();
	void CreateCommandQueue();
	void CreateCommandAllocators();
	void CreateFence();
	void CreateSwapChain();
	void CreateCommandList();

	void ResetCommandList() const;

	void CreateRTVDescriptorHeap();
	void CreateRenderTargetViews();

	void CreateDepthStencilView();
	void CreateDSVDescriptorHeap();
	
	void CreateVertexAndIndexBuffer();

	void CreateDescriptorHeaps();
	void CreateWorldProjCBV();

	void CreateRasterizationRootSignature();

	void CreatePipelineState();

	void SetViewport();
	void SetScissorRect();

	void FlushCommandQueue();
	
	bool CheckTearingSupport();
	bool CheckRayTracingSupport() const;

	void SetupAccelerationStructures();

	void InitializeImGui();
	void RenderImGui();
	
	std::shared_ptr<RaytracePass> m_raytracePass;
	std::shared_ptr<AccelerationStructures> m_accelerationStructures;
	
	bool m_tearingSupport = false;
	bool m_rasterize = true;
	
	Microsoft::WRL::ComPtr<IDXGIAdapter4> GetHardwareAdapter(bool useWarp = false);
	Microsoft::WRL::ComPtr<ID3D12Device5> GetDeviceForAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter);

	Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(const void* initData, UINT64 byteSize, Microsoft::WRL::ComPtr<ID3D12Resource> &uploadBuffer);

	Microsoft::WRL::ComPtr<ID3D12InfoQueue> m_infoQueue;
	
	Microsoft::WRL::ComPtr<ID3D12Device5> m_d3d12Device;
	Microsoft::WRL::ComPtr<IDXGIFactory6> m_dxgiFactory;
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_d3d12CommandQueue;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_d3d12CommandAllocators[Constants::Graphics::NUM_FRAMES];
	Microsoft::WRL::ComPtr<ID3D12Fence> m_d3d12Fence;
	Microsoft::WRL::ComPtr<IDXGISwapChain3> m_dxgiSwapChain;
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_d3d12CommandList;

	DXGI_FORMAT m_backBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT m_depthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
	
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_d3d12RTVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_d3d12RenderTargets[Constants::Graphics::NUM_FRAMES];
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
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvCbvUavDescriptorHeap;
	BYTE* m_mappedData = nullptr;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;

	Microsoft::WRL::ComPtr<IDxcBlob> m_pixelShader;
	Microsoft::WRL::ComPtr<IDxcBlob> m_vertexShader;

	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineStateObject;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;
	
	UINT m_frameIndex = 0;
	UINT64 m_fenceValues[Constants::Graphics::NUM_FRAMES] = {};
	HANDLE m_fenceEvent = nullptr;

	UINT m_rtvDescriptorSize = 0;

	int m_lastMousePosX = 0;
	int m_lastMousePosY = 0;
	float m_theta = 1.5f * DirectX::XM_PI;
	float m_phi = DirectX::XM_PIDIV4;
	float m_radius = 5.0f;

	DirectX::XMFLOAT4X4 m_view = Math::Identity4x4();
	DirectX::XMFLOAT4X4 m_world = Math::Identity4x4();
	DirectX::XMFLOAT4X4 m_proj = Math::Identity4x4();
	DirectX::XMMATRIX m_worldViewProj = DirectX::XMMatrixIdentity();

	std::shared_ptr<Primitive> m_primitive;
};
