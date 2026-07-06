#pragma once
#include <vector>

#include "Constants.h"
#include "Headless.h" // HeadlessConfig
#include "InputElements.h"
#include "RasterDebugMode.h" // VxpgStage
#include "SceneResources/LightData.h"
#include "Keyboard.h"
#include "SimpleMath.h"
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
class VBufferPass;
class LightInjectionPass;
class VoxelGuidingBuildPass;
class SupervoxelClusterPass;
class SuperpixelBuildPass;
class FrameAccumulationPass;
class PostProcessPass;
class AccelerationStructures;
class ScreenshotManager;
class PlacesManager;
struct ScreenshotMetadata;
class VoxelizationPass;



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
	void OnMouseWheel(int delta);
	void OnKeyDown(unsigned long long btnState) const;
	
	void ToggleRasterization();
	void ExecuteCommandsAndReset();

	// Setup verbs shared by the interactive UI callbacks and the headless runner.
	void LoadScene(const std::wstring& path);
	bool SetTechnique(const std::string& name);
	void SetTechniqueByIndex(int index);
	void SetRaytracing(bool enabled) { m_rasterize = !enabled; }
	void SetHeadless(bool headless) { m_headless = headless; }
	void ApplyRenderConfig(const HeadlessConfig& config);
	void SetLights(const std::vector<LightData>& lights);

	std::vector<std::string> GetTechniqueNames() const;
	std::vector<std::string> GetPlaceNames() const;
	bool GoToPlace(const std::string& name);

	// Arm a capture writing to dir/stem (empty => default screenshots dir / auto name).
	void ArmScreenshot(float seconds, const std::string& model, const std::string& place,
	                   const std::string& outDir, const std::string& stem);
	bool ScreenshotIdle() const;

	std::pair<std::shared_ptr<VertexBuffer>, std::shared_ptr<IndexBuffer>> Renderer::CreateSceneResources(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);
	std::shared_ptr<Texture> CreateTextureFromGLTF(const tinygltf::Image& image);
	std::shared_ptr<GameObject> InstantiateGameObject();

	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> GetCommandList() const { return m_d3d12CommandList; }

	template <typename T>
	std::shared_ptr<StructuredBuffer<T>> CreateStructuredBuffer(const std::vector<T> &data); 
	
	inline static Microsoft::WRL::ComPtr<ID3D12Device5> g_device;
	inline static int g_textureIndex = 0;

	static std::array<const CD3DX12_STATIC_SAMPLER_DESC, Constants::Graphics::STATIC_SAMPLERS_COUNT> GetStaticSamplers();
	
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

	void InitializeEditorUI();

	void OnShaderReload();
	void LoadSkybox(const std::wstring& path);
	void WriteVoxelUavsToGlobalHeap();
	void WriteSuperpixelUavsToGlobalHeap();
	void WireGuidingResources();
	// Runs the linear VXPG pipeline (voxelize -> inject -> guiding build ->
	// supervoxel cluster) up to and including the requested stage.
	void RunVxpgPipelineUpTo(VxpgStage stage);

	std::shared_ptr<RaytracePass> m_raytracePass;
	std::shared_ptr<VBufferPass> m_vbufferPass;
	std::shared_ptr<LightInjectionPass> m_lightInjectionPass;
	std::shared_ptr<FrameAccumulationPass> m_accumulationPass;
	std::shared_ptr<PostProcessPass> m_postProcessPass;
	std::shared_ptr<ScreenshotManager> m_screenshotManager;
	std::vector<std::shared_ptr<AccelerationStructures>> m_accelerationStructures;

	DirectX::SimpleMath::Vector3 m_prevCameraPos = {};
	DirectX::XMFLOAT4            m_prevCameraRot = { 0, 0, 0, 1 };

	bool m_tearingSupport = false;
	bool m_rasterize = true;
	bool m_headless = false;
	int  m_activeTechniqueIndex = 0;
	
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

	DirectX::XMFLOAT4X4 m_world = MathUtils::XMFloat4x4Identity();

	std::shared_ptr<Camera> m_camera;

	std::shared_ptr<DirectX::Keyboard::KeyboardStateTracker> m_keyboardTracker;

	int m_currentModelCBVIndex = 0;
	
	std::shared_ptr<Scene> m_scene;

	std::shared_ptr<Material> m_material;
	std::vector<std::shared_ptr<Texture>> m_textures = std::vector<std::shared_ptr<Texture>>();

	std::shared_ptr<PassConstants> m_passConstants;
	std::shared_ptr<StructuredBuffer<float>> m_randomBuffer;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_skyboxResource;

	std::shared_ptr<class EditorUI> m_editorUI;
	std::shared_ptr<PlacesManager> m_placesManager;
	std::shared_ptr<VoxelizationPass> m_voxelizationPass;
	std::shared_ptr<VoxelGuidingBuildPass> m_voxelGuidingBuildPass;
	std::shared_ptr<SupervoxelClusterPass> m_supervoxelClusterPass;
	std::shared_ptr<SuperpixelBuildPass> m_superpixelBuildPass;

	ScreenshotMetadata BuildScreenshotMetadata(const std::string& modelName, const std::string& placeName) const;
};

template <typename T>
std::shared_ptr<StructuredBuffer<T>> Renderer::CreateStructuredBuffer(const std::vector<T>& data)
{
	const auto data_size = sizeof(T) * data.size();
	auto buffer_size = std::max(data_size, sizeof(T)); // D3D12 disallows 0-byte resources

	Microsoft::WRL::ComPtr<ID3D12Resource> upload_buffer;

	auto cpuData = static_cast<BYTE*>(malloc(buffer_size));
	memset(cpuData, 0, buffer_size);
	if (data_size > 0)
		memcpy(cpuData, data.data(), data_size);

	Microsoft::WRL::ComPtr<ID3D12Resource> default_buffer = RenderingUtils::CreateDefaultBuffer(g_device.Get(), m_d3d12CommandList.Get(), cpuData, buffer_size, upload_buffer);
	auto structured_buffer = std::make_shared<StructuredBuffer<T>>(g_device, default_buffer, data.size());

	const std::string type_name = typeid(T).name();
	const std::string resource_name = "Structured Buffer " + type_name;
	
	structured_buffer->SetResourceName(resource_name);

	ExecuteCommandsAndReset();
	AssertFreeClear(&cpuData);

	return structured_buffer;
}
