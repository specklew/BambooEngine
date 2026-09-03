#pragma once
#include <vector>

#include "Constants.h"
#include "GlobalDescriptorHeap.h"
#include "GraphicsDevice.h"
#include "RenderGraph.h"
#include "RenderTechnique.h"
#include "RootSignatureLibrary.h"
#include "Headless.h" // HeadlessConfig
#include "InputElements.h"
#include "RasterDebugMode.h"
#include "SceneResources/LightData.h"
#include "Keyboard.h"
#include "SimpleMath.h"
#include "Resources/StructuredBuffer.h"
#include "Resources/Texture.h"
#include "Utils/Utils.h"

class IndexBuffer;
class VertexBuffer;
class PassConstants;
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
class VBufferPass;
class LightInjectionPass;
class VoxelGuidingBuildPass;
class VxpgFingerprintPass;
class VxpgClusterPass;
class VxpgClusterVisibilityPass;
class VxpgLightTreePass;
class SuperpixelBuildPass;
class FrameAccumulationPass;
class PostProcessPass;
class DebugViewPass;
class AccelerationStructures;
class ScreenshotManager;
class StatesManager;
struct ScreenshotMetadata;
struct CaptureSchedule;
struct WarmUpReport;
class GpuMemoryReport;
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
	
	void ExecuteCommandsAndReset();

	// Setup verbs shared by the interactive UI callbacks and the headless runner.
	// statesKey overrides the states.json key derived from the file name. Research
	// scenes rename the file (veach-ajar/veach_ajar_core.glb) but share one camera
	// set, so the folder- or user-supplied name is the scene's real identity.
	void LoadScene(const std::wstring& path, const std::string& statesKey = {});
	bool SetTechnique(const std::string& name);
	void SetTechniqueByIndex(int index);
	void SetHeadless(bool headless) { m_headless = headless; }
	void ApplyRenderConfig(const HeadlessConfig& config);
	void SetLights(const std::vector<LightData>& lights);

	// Per-node GPU timings accumulate across frames; a technique switch makes the
	// history meaningless, so the headless runner clears it between techniques.
	void ResetGraphTimingHistory() { m_renderGraph.ResetTimingHistory(); }

	std::vector<std::string> GetTechniqueNames() const;
	std::vector<std::string> GetStateNames() const;

	// The active technique's own debug views, and the selector headless drives to
	// walk them. Indices belong to that technique's enumeration.
	std::vector<RenderTechnique::DebugView> GetTechniqueDebugViews() const;
	bool SetTechniqueDebugView(int index);
	bool GoToState(const std::string& name);

	// Arm a capture writing to dir/stem (empty => default screenshots dir / auto name).
	void ArmScreenshot(float seconds, const std::string& model, const std::string& place,
	                   const std::string& outDir, const std::string& stem);
	// Benchmark form: a full schedule (frame or time budget, checkpoints) plus the
	// provenance a measurement has to carry to be reproducible.
	void ArmScreenshot(const CaptureSchedule& schedule, const std::string& model, const std::string& place,
	                   const std::string& outDir, const std::string& stem,
	                   uint32_t imageIndex, uint32_t imageCount, const WarmUpReport& warmup);
	bool ScreenshotIdle() const;
	// The settings point every subsequent capture belongs to (--cvar-matrix). Held on
	// the renderer rather than passed per capture because it is provenance, not an
	// argument: whatever is set here is what the sidecar reports it measured.
	void SetSettingsTag(const std::string& tag) { m_settingsTag = tag; }
	// Idle means the accumulation window is over, not that the PNG is on disk —
	// encoding is asynchronous. A run must drain the queue before it exits.
	void WaitForScreenshotWrites() const;

	std::pair<std::shared_ptr<VertexBuffer>, std::shared_ptr<IndexBuffer>> CreateSceneResources(
		const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);
	std::shared_ptr<Texture> CreateTextureFromGLTF(const tinygltf::Image& image);
	std::shared_ptr<GameObject> InstantiateGameObject();

	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> GetCommandList() const { return m_d3d12CommandList; }

	template <typename T>
	std::shared_ptr<StructuredBuffer<T>> CreateStructuredBuffer(const std::vector<T> &data); 
	
	inline static Microsoft::WRL::ComPtr<ID3D12Device5> g_device;

	// Headless (benchmark) runs disable the D3D12 debug layer. It also switches on
	// GPU-based validation (ENABLE_GPU_BASED_VALIDATION, GraphicsDevice.cpp), and
	// the pair is not a tax a measurement can absorb: the inline-RQ integrator on
	// veach-ajar Deep Light goes 5.1 ms -> 1118.5 ms per frame with it on (measured
	// 2026-08-23). It also taxes DispatchRays and Dispatch unevenly, so it skews
	// equal-time comparisons on top of slowing them. Must be set before
	// Initialize(). Interactive Debug builds keep the layer.
	inline static bool g_enableDebugLayer = true;

	static std::array<const CD3DX12_STATIC_SAMPLER_DESC, Constants::Graphics::STATIC_SAMPLERS_COUNT> GetStaticSamplers();
	
private:
	void CreateCommandList();

	void ResetCommandList() const;

	void CreateRTVDescriptorHeap();
	void CreateRenderTargetViews();

	void CreateDepthStencilView();
	void CreateDSVDescriptorHeap();

	void CreateWorldProjCBV();

	void SetViewport();
	void SetScissorRect();

	void FlushCommandQueue();

	void CreateTextureSRV(const std::shared_ptr<Texture>& texture);
	void CreateVertexSRV();
	void CreateIndexSRV();

	void InitializeEditorUI();

	void OnShaderReload();
	void LoadSkybox(const std::wstring& path);

	// Appended after the active technique's own nodes, when it rendered offscreen.
	// Everything that touches the back buffer is a node, so one Compile()/Execute()
	// covers the whole frame whichever technique produced it.
	void BuildDisplayChain(GraphResourceHandle techniqueOutput, Texture& techniqueOutputTexture,
	                       GraphResourceHandle backBufferHandle, Texture& backBuffer);
	// Replaces the display chain outright while a buffer debug view is selected.
	void BuildBufferDebugChain(GraphResourceHandle backBufferHandle, Texture& backBuffer);
	void BindBackBufferTarget(uint32_t frameIndex) const;

	void DumpRenderGraphIfRequested();
	void WriteVoxelUavsToGlobalHeap();
	void WriteSuperpixelUavsToGlobalHeap();
	void WriteClusterVisibilityUavsToGlobalHeap();
	// Hands a freshly created technique the renderer-owned objects its factory
	// could not take. Runs before Initialize: the raster pipeline state bakes in
	// the frame target formats.
	void WireTechniqueResources(const std::shared_ptr<RenderTechnique>& technique);
	// Adds every VXPG stage to the frame's graph. Which of them survive is the
	// graph's decision: a stage nothing reads is culled.
	void BuildVxpgGraph();
	bool FrameUsesVoxelGuiding() const;

	VxpgGraphHandles m_vxpg;
	// Latched from vxpg.cluster.dumpStats at graph-build time so the copy node,
	// the frame's flush and the readback all agree on one answer.
	bool m_clusterStatsPending = false;
	bool m_guidingProbePending = false;
	// The unsteerable-share accumulators are filled by the INTEGRATOR, so their readback
	// node has to sit past the technique rather than inside the guiding chain.
	bool m_unsteerableProbePending = false;
	// Start of the current frame, for the in-frame duration AccountFrame consumes.
	std::chrono::steady_clock::time_point m_frameStart;

	std::shared_ptr<RenderTechnique> m_technique;
	std::shared_ptr<VBufferPass> m_vbufferPass;
	std::shared_ptr<LightInjectionPass> m_lightInjectionPass;
	std::shared_ptr<FrameAccumulationPass> m_accumulationPass;
	std::shared_ptr<PostProcessPass> m_postProcessPass;
	std::shared_ptr<DebugViewPass> m_debugViewPass;
	std::shared_ptr<ScreenshotManager> m_screenshotManager;
	std::vector<std::shared_ptr<AccelerationStructures>> m_accelerationStructures;

	DirectX::SimpleMath::Vector3 m_prevCameraPos = {};
	DirectX::XMFLOAT4            m_prevCameraRot = { 0, 0, 0, 1 };

	bool m_headless = false;
	int  m_activeTechniqueIndex = 0;
	std::string m_settingsTag; // --cvar-matrix point the next captures belong to

	std::unique_ptr<GraphicsDevice> m_graphicsDevice;
	RenderGraph                     m_renderGraph;
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_d3d12CommandList;

	DXGI_FORMAT m_backBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT m_depthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
	
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_d3d12RTVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_d3d12RenderTargets[Constants::Graphics::NUM_FRAMES];
	// State-tracked wrappers over the swap-chain buffers; all barriers go through these
	std::unique_ptr<Texture> m_backBufferTextures[Constants::Graphics::NUM_FRAMES];
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_d3d12DSVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_depthStencilBuffer;
	std::unique_ptr<Texture> m_depthStencilTexture;
	
	std::shared_ptr<ConstantBuffer> m_modelIndexConstantBuffer;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;
	std::unordered_map<std::shared_ptr<Model>, std::shared_ptr<AccelerationStructureBuffers>> m_modelsBLASes;
	
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
	std::unique_ptr<Texture> m_skyboxTexture;

	std::shared_ptr<class EditorUI> m_editorUI;
	std::shared_ptr<StatesManager> m_statesManager;
	std::shared_ptr<VoxelizationPass> m_voxelizationPass;
	std::shared_ptr<VoxelGuidingBuildPass> m_voxelGuidingBuildPass;
	std::shared_ptr<VxpgFingerprintPass> m_fingerprintPass;
	std::shared_ptr<VxpgClusterPass> m_clusterPass;
	std::shared_ptr<VxpgClusterVisibilityPass> m_clusterVisibilityPass;
	std::shared_ptr<VxpgLightTreePass> m_lightTreePass;
	std::shared_ptr<SuperpixelBuildPass> m_superpixelBuildPass;
	// Re-armed whenever the chain's shape can have changed (technique, scene, resize).
	bool m_gpuMemoryReportPending = true;

	ScreenshotMetadata BuildScreenshotMetadata(const std::string& modelName, const std::string& placeName) const;
	// P5: walks every pass that holds GPU resources. Cheap (a desc query per resource),
	// so it is taken fresh rather than cached — the chain resizes with the grid and with
	// the window, and a cached figure would quietly describe the previous configuration.
	GpuMemoryReport CollectGpuMemory() const;
	void LogGpuMemoryOnce();
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

	Microsoft::WRL::ComPtr<ID3D12Resource> default_buffer = RenderingUtils::CreateDefaultBuffer(
		g_device.Get(), m_d3d12CommandList.Get(), cpuData, buffer_size, upload_buffer);
	auto structured_buffer = std::make_shared<StructuredBuffer<T>>(g_device, default_buffer, data.size());

	const std::string type_name = typeid(T).name();
	const std::string resource_name = "Structured Buffer " + type_name;
	
	structured_buffer->SetResourceName(resource_name);

	ExecuteCommandsAndReset();
	AssertFreeClear(&cpuData);

	return structured_buffer;
}
