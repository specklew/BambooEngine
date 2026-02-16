#pragma once
#include <vector>

#include "Constants.h"
#include "InputElements.h"
#include "Keyboard.h"
#include "Resources/StructuredBuffer.h"
#include "Utils/Utils.h"

class IndexBuffer;
class VertexBuffer;
class PassConstants;
class Texture;
struct Material;
class GameObject;
struct AccelerationStructureBuffers;
class Scene;
class Model;
class ConstantBuffer;

namespace DirectX
{
	class Keyboard;
}

class Camera;
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

	void OnResize();
	void OnMouseMove(unsigned long long btnState, int x, int y);
	void OnKeyDown(unsigned long long btnState) const;
	
	void ToggleRasterization();
	void ExecuteCommandsAndReset();

	std::pair<std::shared_ptr<VertexBuffer>, std::shared_ptr<IndexBuffer>> Renderer::CreateSceneResources(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);
	std::shared_ptr<Texture> CreateTextureFromGLTF(const tinygltf::Image& image);
	std::shared_ptr<GameObject> InstantiateGameObject();

	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> GetCommandList() const { return m_d3d12CommandList; }

	template <typename T>
	std::shared_ptr<StructuredBuffer<T>> CreateStructuredBuffer(const std::vector<T> &data); 
	
	inline static Microsoft::WRL::ComPtr<ID3D12Device5> g_device;
	inline static int g_textureIndex = 0;
	
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

	void CreateDescriptorHeaps();
	void CreateWorldProjCBV();

	void CreateRasterizationRootSignature();

	void CreatePipelineState();

	void SetViewport();
	void SetScissorRect();

	void FlushCommandQueue();
	
	bool CheckTearingSupport();
	bool CheckRayTracingSupport() const;

	void CreateTextureSRV(const std::shared_ptr<Texture>& texture);
	void CreateVertexSRV();
	void CreateIndexSRV();

	void InitializeImGui();
	void RenderImGui();

	void OnShaderReload();

	std::array<const CD3DX12_STATIC_SAMPLER_DESC, Constants::Graphics::STATIC_SAMPLERS_COUNT> GetStaticSamplers();
	
	std::shared_ptr<RaytracePass> m_raytracePass;
	std::vector<std::shared_ptr<AccelerationStructures>> m_accelerationStructures;
	
	bool m_tearingSupport = false;
	bool m_rasterize = true;
	
	Microsoft::WRL::ComPtr<IDXGIAdapter4> GetHardwareAdapter(bool useWarp = false);
	Microsoft::WRL::ComPtr<ID3D12Device5> GetDeviceForAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter);

	Microsoft::WRL::ComPtr<ID3D12InfoQueue> m_infoQueue;
	
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
	
	std::shared_ptr<ConstantBuffer> m_projectionMatrixConstantBuffer;
	std::shared_ptr<ConstantBuffer> m_modelIndexConstantBuffer;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvCbvUavDescriptorHeap;
	BYTE* m_mappedData = nullptr;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;

	Microsoft::WRL::ComPtr<IDxcBlob> m_pixelShader;
	Microsoft::WRL::ComPtr<IDxcBlob> m_vertexShader;

	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineStateObject;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;
	std::unordered_map<std::shared_ptr<Model>, std::shared_ptr<AccelerationStructureBuffers>> m_modelsBLASes;
	
	UINT m_frameIndex = 0;
	UINT64 m_fenceValue = 0;
	HANDLE m_fenceEvent = nullptr;

	UINT m_rtvDescriptorSize = 0;

	int m_lastMousePosX = 0;
	int m_lastMousePosY = 0;
	float m_theta = 1.5f * DirectX::XM_PI;
	float m_phi = DirectX::XM_PIDIV4;
	float m_radius = 5.0f;

	int previousScene;
	
	DirectX::XMFLOAT4X4 m_world = MathUtils::XMFloat4x4Identity();

	std::shared_ptr<Camera> m_camera;

	std::shared_ptr<DirectX::Keyboard::KeyboardStateTracker> m_keyboardTracker;

	int m_currentModelCBVIndex = 0;
	
	std::shared_ptr<Scene> m_scene;

	std::shared_ptr<Material> m_material;
	std::vector<std::shared_ptr<Texture>> m_textures = std::vector<std::shared_ptr<Texture>>();

	std::shared_ptr<PassConstants> m_passConstants;
};

template <typename T>
std::shared_ptr<StructuredBuffer<T>> Renderer::CreateStructuredBuffer(const std::vector<T>& data)
{
	auto buffer_size = sizeof(T) * data.size();

	Microsoft::WRL::ComPtr<ID3D12Resource> upload_buffer;

	auto cpuData = static_cast<BYTE*>(malloc(buffer_size));
	memcpy(cpuData, data.data(), buffer_size);

	Microsoft::WRL::ComPtr<ID3D12Resource> default_buffer = RenderingUtils::CreateDefaultBuffer(g_device.Get(), m_d3d12CommandList.Get(), cpuData, buffer_size, upload_buffer);
	auto structured_buffer = std::make_shared<StructuredBuffer<T>>(g_device, default_buffer, data.size());

	const std::string type_name = typeid(T).name();
	const std::string resource_name = "Structured Buffer " + type_name;
	
	structured_buffer->SetResourceName(resource_name);

	ExecuteCommandsAndReset();
	AssertFreeClear(&cpuData);

	return structured_buffer;
}
