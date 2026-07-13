#include "pch.h"

#include "Renderer.h"

#include <algorithm>
#include <crtdbg.h>

#include "Camera.h"
#include "DDSTextureLoader/DDSTextureLoader12.h"
#include "EditorUI.h"
#include "FrameAccumulationPass.h"
#include "PlacesManager.h"
#include "PostProcessPass.h"
#include "Utils/CVars.h"
#include "Utils/GpuMarker.h"
#include "Utils/Utils.h"
#include "SceneResources/GameObject.h"
#include "imgui.h"
#include "backends/imgui_impl_dx12.h"

#include "InputElements.h"
#include "SceneResources/ModelLoading.h"
#include "SceneResources/Primitive.h"
#include "RaytracePass.h"
#include "Techniques/PathTracingPass.h"
#include "ScreenshotManager.h"
#include "VoxelizationPass.h"
#include "LightInjectionPass.h"
#include "VBufferPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgFingerprintPass.h"
#include "VxpgClusterPass.h"
#include "VxpgClusterVisibilityPass.h"
#include "VxpgLightTreePass.h"
#include "SuperpixelBuildPass.h"
#include "Techniques/GuidedPathTracingPass.h"
#include "RasterDebugMode.h"
#include "RaytraceDebugMode.h"
#include "GuidingDebugView.h"
#include "SceneResources/Scene.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Window.h"
#include "Resources/ConstantBuffer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/Texture.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/Material.h"
#include "SceneResources/Model.h"
#include "tinygltf/tiny_gltf.h"
#include "Utils/PassConstants.h"

#include <filesystem>

#include "AccelerationStructures.h"

#ifdef _DEBUG
#define ENABLE_GPU_BASED_VALIDATION 1
// Routes every D3D12 debug-layer message to spdlog so validation errors/warnings
// appear in the engine console (and the headless log), not just the attached
// debugger's output window.
static void CALLBACK D3D12DebugMessageCallback(
	D3D12_MESSAGE_CATEGORY /*category*/, D3D12_MESSAGE_SEVERITY severity,
	D3D12_MESSAGE_ID /*id*/, LPCSTR description, void* /*context*/)
{
	switch (severity)
	{
	case D3D12_MESSAGE_SEVERITY_CORRUPTION:
	case D3D12_MESSAGE_SEVERITY_ERROR:
		spdlog::error("[D3D12] {}", description);
		break;
	case D3D12_MESSAGE_SEVERITY_WARNING:
		spdlog::warn("[D3D12] {}", description);
		break;
	default:
		spdlog::debug("[D3D12] {}", description);
		break;
	}
}
#endif

using namespace Microsoft::WRL;

namespace
{
    std::string ExtractModelName(const std::string& path)
    {
        return ToLowerAscii(std::filesystem::path(path).stem().string());
    }
    std::string ExtractModelName(const std::wstring& path)
    {
        return ToLowerAscii(std::filesystem::path(path).stem().string());
    }
}

static AutoCVarFloat g_cameraSpeed("renderer.camera.speed", "Specifies the base speed of camera", 1.0f, CVarFlags::EditDrag, 0.1f, 100.0f);
static AutoCVarFloat g_cameraScrollFactor("renderer.camera.scrollFactor", "Multiplier per scroll tick for camera speed", 1.2f, CVarFlags::EditDrag, 1.01f, 3.0f);
static AutoCVarFloat g_uvCoordX("renderer.uv.x", "Texture uv x offset", 0.0f, CVarFlags::EditDrag, 0.0f, 1.0f);
static AutoCVarFloat g_uvCoordY("renderer.uv.y", "Texture uv y offset", 0.0f, CVarFlags::EditDrag, 0.0f, 1.0f);
static AutoCVarEnum g_rasterizationDebugMode("renderer.rasterDebugMode", "Rasterization shader debug visualization mode", RasterDebugMode::None,
                                             CVarFlags::None, FormatDebugViewDocs<RasterDebugMode>(kRasterDebugModeDocs));
static AutoCVarEnum g_raytraceDebugMode("renderer.raytraceDebugMode", "Raytracing shader debug visualization mode", RaytraceDebugMode::None,
                                        CVarFlags::None, FormatDebugViewDocs<RaytraceDebugMode>(kRaytraceDebugModeDocs));
static AutoCVarFloat3 g_cameraPos("renderer.camera.position", "Camera world position", {0.0f, 0.0f, -10.0f});
static AutoCVarFloat3 g_cameraRot("renderer.camera.rotation", "Camera rotation (pitch, yaw, roll) degrees", {0.0f, 0.0f, 0.0f});
static AutoCVarInt g_numSamplesPerPixel("renderer.samplesPerPixel", "Number of samples per pixel", 1, CVarFlags::EditDrag, 1, 64);
static AutoCVarInt g_numBounces("renderer.numBounces", "Number of bounces", 1, CVarFlags::EditDrag, 0, 7);
static AutoCVarInt   g_accumulationEnabled("renderer.accumulation.enabled","Enable temporal frame accumulation when camera is still", 0, CVarFlags::EditCheckbox);
static AutoCVarFloat g_exposure("renderer.postprocess.exposure","Exposure multiplier applied before display", 1.0f, CVarFlags::EditDrag, 0.0f, 10.0f);
static AutoCVarFloat g_contrast("renderer.postprocess.contrast", "Pre-ACES contrast power curve", 1.0f, CVarFlags::EditDrag, 0.1f, 3.0f);
static AutoCVarFloat g_saturation("renderer.postprocess.saturation", "Post-ACES saturation", 1.0f, CVarFlags::EditDrag, 0.0f, 2.0f);
static AutoCVarFloat g_lift("renderer.postprocess.lift", "Post-ACES shadow lift", 0.0f, CVarFlags::EditDrag, 0.0f, 0.5f);
static AutoCVarInt   g_voxelGridDim("voxel.gridDim", "Voxel grid resolution (one axis)", 64, CVarFlags::EditDrag, 32, 256);
static AutoCVarFloat g_voxelAabbPad("voxel.aabbPadCells", "Voxel grid padding in cells (unused V1)", 0.5f, CVarFlags::EditDrag, 0.0f, 4.0f);
static AutoCVarInt   g_voxelInjectUseAvg("voxel.inject.useAvg", "Injection accumulation: 1 = average (add + count), 0 = max", 1, CVarFlags::EditCheckbox);
// Default ON (deviation from SIByL's shipped default): full-cube bounds made the
// guided sampler aim at mostly-empty cube space — view-4 acceptance was ~red
// everywhere; tight bounds turned the lit half of Sponza green (2026-07-09).
static AutoCVarInt   g_voxelBakeUseCompact("voxel.bake.useCompact", "Bake tight per-voxel triangle AABBs instead of full cubes (SIByL default: off)", 1, CVarFlags::EditCheckbox);
static AutoCVarInt   g_voxelBakeClipping("voxel.bake.clipping", "Clip triangles against the voxel cube before the tight AABB (SIByL default: off)", 0, CVarFlags::EditCheckbox);
// Jitter ON is a deliberate deviation (SIByL uses pixel centers): the PT
// reference anti-aliases its primaries, so a pixel-center VBuffer leaves a
// constant silhouette mismatch vs the reference (measured 2026-07-10: RMSE
// 0.0180 vs 0.0117 with jitter, same frame count) — edge error, not variance.
static AutoCVarInt   g_vbufferJitter("vxpg.vbufferJitter", "Sub-pixel jitter for the shared VBuffer primaries (off = SIByL-literal pixel-center, no edge AA)", 1, CVarFlags::EditCheckbox);
static AutoCVarFloat g_superpixelWeight("superpixel.weight", "SLIC coherence weight: screen-xy vs world-position", 0.6f, CVarFlags::EditDrag, 0.0f, 4.0f);
static AutoCVarFloat g_superpixelPosNormalizer("superpixel.posNormalizer", "SLIC world-position distance normalizer (squared)", 8.3329f, CVarFlags::EditDrag, 0.001f, 1000.0f);
static AutoCVarFloat g_voxelHeatScale("voxel.heatScale", "Irradiance heat map scale", 1.0f, CVarFlags::EditDrag, 0.001f, 100.0f);
// Default = power: the ported integrator (vxguiding-gi strategy 5) hardcodes
// power-heuristic squaring; balance stays available as a variance experiment.
static AutoCVarInt   g_guidingPowerMis("guiding.powerMis", "MIS heuristic: 0 = balance, 1 = power", 1, CVarFlags::EditCheckbox);
static AutoCVarFloat g_indirectSkyClamp("pathtracing.indirectSkyClamp", "Clamp indirect-bounce skybox radiance to suppress HDR-sun fireflies for benchmark convergence. 0 = disabled (unbiased)", 0.0f, CVarFlags::EditDrag, 0.0f, 1000.0f);
static AutoCVarInt   g_skyLighting("pathtracing.skyLighting", "Skybox radiance lights surfaces via indirect rays; 0 = sky is background-only (benchmark isolation: the VXPG guide only targets direct-lit surfaces)", 1, CVarFlags::EditCheckbox);
static AutoCVarEnum  g_guidingDebugView("guiding.debugView", "Guided PT debug visualization", GuidingDebugView::None,
                                        CVarFlags::None, FormatDebugViewDocs<GuidingDebugView>(kGuidingDebugViewDocs));

void Renderer::Initialize()
{
	spdlog::info("Initializing renderer...");

	m_camera = std::make_shared<Camera>();
	m_keyboardTracker = std::make_shared<DirectX::Keyboard::KeyboardStateTracker>();

	m_placesManager = std::make_shared<PlacesManager>();
	m_placesManager->Load();
	m_placesManager->SetCamera(*m_camera);
	
#ifdef _DEBUG
	// Route CRT assertion failures to stderr instead of a modal dialog, so
	// asserts appear in the console / headless log rather than hanging the run.
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

	SetupDeviceAndDebug();
	CheckTearingSupport();
	
	if (!CheckRayTracingSupport()) 	throw std::runtime_error("Raytracing is not supported on this device.");;
	
	CreateCommandQueue();
	CreateCommandAllocators();
	CreateFence();
	CreateSwapChain();
	
	CreateCommandList();
	ResetCommandList();
	
	CreateRTVDescriptorHeap();
	CreateRenderTargetViews();

	CreateDSVDescriptorHeap();
	CreateDepthStencilView();

	// Execute depth stencil creation commands
	ExecuteCommandsAndReset();
	// Finish execution - reset command list for next setup commands

	CreateDescriptorHeaps();
	CreateWorldProjCBV();

	m_scene = ModelLoading::LoadScene(*this, AssetId("resources/models/abeautifulgame.glb"));
	m_placesManager->OnSceneChanged(ExtractModelName(std::string("resources/models/abeautifulgame.glb")));

	CreateVertexSRV();
	CreateIndexSRV();

	m_material = std::make_shared<Material>();
	m_material->m_data.baseColorFactor = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);

	m_passConstants = std::make_shared<PassConstants>();
	
	CreateRasterizationRootSignature();

	CreatePipelineState();

	ExecuteCommandsAndReset();

	std::vector<float> randomData(3840 * 2190);
	for (int i = 0; i < 3840 * 2190; ++i)
	{
		randomData[i] = RaytracerRandom::g_random->GetRandomFloat();
	}
	
	m_randomBuffer = CreateStructuredBuffer<float>(randomData);
	const auto& registry = RaytracePass::GetRegistry();
	// Default to vanilla path tracing regardless of static-init registration order
	auto defaultEntry = std::find_if(registry.begin(), registry.end(),
		[](const TechniqueEntry& e) { return e.name == "Path Tracing"; });
	if (defaultEntry == registry.end() && !registry.empty())
		defaultEntry = registry.begin();
	m_raytracePass = (defaultEntry == registry.end()) ? std::make_shared<RaytracePass>() : defaultEntry->create();
	m_raytracePass->Initialize(g_device, m_d3d12CommandList, m_scene, m_srvCbvUavDescriptorHeap, m_randomBuffer->GetUnderlyingResource(), m_passConstants);

	m_accumulationPass = std::make_shared<FrameAccumulationPass>();
	m_accumulationPass->Initialize(g_device, m_d3d12CommandList);

	m_postProcessPass = std::make_shared<PostProcessPass>();
	m_postProcessPass->Initialize(g_device, m_d3d12CommandList);

	m_screenshotManager = std::make_shared<ScreenshotManager>();
	m_screenshotManager->Initialize(g_device, m_d3d12CommandList);

	m_voxelizationPass = std::make_shared<VoxelizationPass>();
	m_voxelizationPass->Initialize(g_device, m_d3d12CommandList, m_rootSignature);

	WriteVoxelUavsToGlobalHeap();

	m_voxelizationPass->OnSceneLoaded(*m_scene);

	m_vbufferPass = std::make_shared<VBufferPass>();
	m_vbufferPass->Initialize(g_device, m_d3d12CommandList, m_scene, m_srvCbvUavDescriptorHeap, m_randomBuffer->GetUnderlyingResource(), m_passConstants);

	m_lightInjectionPass = std::make_shared<LightInjectionPass>();
	m_lightInjectionPass->SetVoxelizationPass(m_voxelizationPass);
	m_lightInjectionPass->Initialize(g_device, m_d3d12CommandList, m_scene, m_srvCbvUavDescriptorHeap, m_randomBuffer->GetUnderlyingResource(), m_passConstants);

	m_voxelGuidingBuildPass = std::make_shared<VoxelGuidingBuildPass>();
	m_voxelGuidingBuildPass->Initialize(g_device, m_d3d12CommandList, m_voxelizationPass);

	m_fingerprintPass = std::make_shared<VxpgFingerprintPass>();
	m_fingerprintPass->Initialize(g_device, m_d3d12CommandList, m_voxelGuidingBuildPass, m_lightInjectionPass);
	m_fingerprintPass->OnResize(Window::Get().GetWidth(), Window::Get().GetHeight());

	m_clusterPass = std::make_shared<VxpgClusterPass>();
	m_clusterPass->Initialize(g_device, m_d3d12CommandList, m_voxelizationPass,
		m_voxelGuidingBuildPass, m_fingerprintPass);

	m_superpixelBuildPass = std::make_shared<SuperpixelBuildPass>();
	m_superpixelBuildPass->Initialize(g_device, m_d3d12CommandList);
	m_superpixelBuildPass->OnResize(Window::Get().GetWidth(), Window::Get().GetHeight(),
		m_lightInjectionPass->GetShadingPointsTexture().Get());
	WriteSuperpixelUavsToGlobalHeap();

	m_clusterVisibilityPass = std::make_shared<VxpgClusterVisibilityPass>();
	m_clusterVisibilityPass->Initialize(g_device, m_d3d12CommandList, m_srvCbvUavDescriptorHeap,
		m_voxelizationPass, m_voxelGuidingBuildPass, m_clusterPass, m_superpixelBuildPass);
	m_clusterVisibilityPass->SetScene(m_scene);
	m_clusterVisibilityPass->OnResize(Window::Get().GetWidth(), Window::Get().GetHeight());
	WriteClusterVisibilityUavsToGlobalHeap();

	m_lightTreePass = std::make_shared<VxpgLightTreePass>();
	m_lightTreePass->Initialize(g_device, m_d3d12CommandList, m_srvCbvUavDescriptorHeap,
		m_voxelizationPass, m_voxelGuidingBuildPass, m_clusterPass, m_clusterVisibilityPass);
	m_lightTreePass->OnResize(Window::Get().GetWidth(), Window::Get().GetHeight());

	WireGuidingResources();

	spdlog::info("Renderer initialized successfully.");

	LoadSkybox(L"Resources/Textures/qwantani_dusk_2_puresky_2k.dds");

	InitializeEditorUI();

	ExecuteCommandsAndReset();
}
float speedMultiplier = 1.0f;
void Renderer::Update(double elapsedTime, double totalTime)
{
	// Apply CVar edits from ImGui only when the CVar value actually changed since last frame.
	// Applying every frame would stomp camera state set via other paths (e.g. PlacesManager::GoTo).
	// Headless drives the camera solely through GoToPlace, so this sync is skipped there.
	if (!m_headless)
	{
	    static bool s_init = false;
	    static DirectX::XMFLOAT3 s_prevCvarPos = {};
	    static DirectX::XMFLOAT3 s_prevCvarRot = {};
	    const DirectX::XMFLOAT3 curCvarPos = g_cameraPos.Get();
	    const DirectX::XMFLOAT3 curCvarRot = g_cameraRot.Get();
	    if (!s_init)
	    {
	        m_camera->SetPosition(curCvarPos);
	        m_camera->SetRotation(DirectX::SimpleMath::Quaternion::CreateFromYawPitchRoll(
	                DirectX::XMConvertToRadians(curCvarRot.y),
	                DirectX::XMConvertToRadians(curCvarRot.x),
	                DirectX::XMConvertToRadians(curCvarRot.z)));
	        s_init = true;
	    }
	    else
	    {
            if (curCvarPos.x != s_prevCvarPos.x || curCvarPos.y != s_prevCvarPos.y || curCvarPos.z != s_prevCvarPos.z)
                    m_camera->SetPosition(curCvarPos);
            if (curCvarRot.x != s_prevCvarRot.x || curCvarRot.y != s_prevCvarRot.y || curCvarRot.z != s_prevCvarRot.z)
                    m_camera->SetRotation(DirectX::SimpleMath::Quaternion::CreateFromYawPitchRoll(
                            DirectX::XMConvertToRadians(curCvarRot.y),
                            DirectX::XMConvertToRadians(curCvarRot.x),
                            DirectX::XMConvertToRadians(curCvarRot.z)));
	    }
	    s_prevCvarPos = curCvarPos;
	    s_prevCvarRot = curCvarRot;
	}

	auto key_state = DirectX::Keyboard::Get().GetState();

	if (key_state.LeftShift || key_state.RightShift)
	{
		if (speedMultiplier < 100.0f) speedMultiplier *= 1.002f;
	}
	else
	{
		speedMultiplier = 1.0f;
	}
	
	if (key_state.W)
	{
		m_camera->AddPosition(m_camera->GetForward() * static_cast<float>(elapsedTime) * g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.S)
	{
		m_camera->AddPosition(m_camera->GetForward() * static_cast<float>(elapsedTime) * -g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.D)
	{
		m_camera->AddPosition(m_camera->GetRight() * static_cast<float>(elapsedTime) * g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.A)
	{
		m_camera->AddPosition(m_camera->GetRight() * static_cast<float>(elapsedTime) * -g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.F2)
	{
		OnShaderReload();
	}

	// Sync camera state back to CVars (camera → CVar)
	{
		const auto& pos = m_camera->GetPosition();
		g_cameraPos.Set({ pos.x, pos.y, pos.z });
		g_cameraRot.Set(m_camera->GetEulerDegrees());
	}

	using namespace DirectX;

    //SimpleMath::Matrix world = m_world;
	SimpleMath::Matrix viewProjection = XMLoadFloat4x4(&m_camera->GetViewProjectionMatrix());
	SimpleMath::Matrix view = XMLoadFloat4x4(&m_camera->GetViewMatrix());
	SimpleMath::Matrix viewProj = XMLoadFloat4x4(&m_camera->GetViewProjectionMatrix());
	SimpleMath::Matrix projection = XMLoadFloat4x4(&m_camera->GetProjectionMatrix());
	
	XMVECTOR det;

	struct ObjectConstants
	{
		XMFLOAT4X4 ViewProj = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 View = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 Projection = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 ViewInverse = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 ProjectionInverse = MathUtils::XMFloat4x4Identity();
	};
	
	ObjectConstants constants;
	XMStoreFloat4x4(&constants.ViewProj, XMMatrixTranspose(viewProjection));
	XMStoreFloat4x4(&constants.View, XMMatrixTranspose(view));
	XMStoreFloat4x4(&constants.Projection, XMMatrixTranspose(projection));
	XMStoreFloat4x4(&constants.ViewInverse, XMMatrixInverse(&det, view));
	XMStoreFloat4x4(&constants.ProjectionInverse, XMMatrixInverse(&det, projection));
	
	memcpy(&m_mappedData[0], &constants, sizeof(constants));

	m_raytracePass->Update(elapsedTime, totalTime);

	m_passConstants->data.uvCoordX = g_uvCoordX.Get();
	m_passConstants->data.uvCoordY = g_uvCoordY.Get();
	m_passConstants->data.debugMode = m_rasterize
		? static_cast<int>(g_rasterizationDebugMode.Get())
		: static_cast<int>(g_raytraceDebugMode.Get());
	m_passConstants->data.numBounces = g_numBounces.Get();
	m_passConstants->data.numSamplesPerPixel = g_numSamplesPerPixel.Get();
	m_passConstants->data.frameIndex++;
	m_passConstants->data.guidingFlags =
		((g_guidingPowerMis.Get() != 0) ? 1u : 0u) |
		((static_cast<uint32_t>(g_guidingDebugView.Get()) & 15u) << 1);
	static_assert(static_cast<int>(GuidingDebugView::SelectedClusterView) <= 15, "GuidingDebugView must fit in 4 bits of guidingFlags");
	const auto& camPos = m_camera->GetPosition();
	m_passConstants->data.cameraWorldPos = { camPos.x, camPos.y, camPos.z };
	m_passConstants->data.numLights = m_scene->GetLightDataBuffer()->GetElementsCount();
	// Per-pixel jitter is derived in-shader from (pixel, frameIndex); the CB
	// just carries the on/off switch.
	m_passConstants->data.vbufferJitterEnabled = (g_vbufferJitter.Get() != 0) ? 1u : 0u;
	m_passConstants->data.indirectSkyClamp = g_indirectSkyClamp.Get();
	m_passConstants->data.skyLightingEnabled = (g_skyLighting.Get() != 0) ? 1u : 0u;
	m_passConstants->Map();

	if (m_scene->IsLightDataDirty())
	{
		m_scene->SetLightDataBuffer(CreateStructuredBuffer(m_scene->GetLightDataCPU()));
		m_scene->ClearLightDataDirty();
	}

	// Camera change detection for accumulation reset. Skipped in headless: the
	// camera only changes between captures (GoToPlace), and ArmScreenshot owns the
	// reset — letting this fire would cancel the pending capture in Tick.
	if (g_accumulationEnabled.Get() && !m_headless)
	{
		auto pos = m_camera->GetPosition();
		auto rot = m_camera->GetRotation();
		bool cameraChanged =
			(pos.x != m_prevCameraPos.x || pos.y != m_prevCameraPos.y || pos.z != m_prevCameraPos.z) ||
			(rot.x != m_prevCameraRot.x || rot.y != m_prevCameraRot.y ||
			 rot.z != m_prevCameraRot.z || rot.w != m_prevCameraRot.w);
		if (cameraChanged)
			m_accumulationPass->Reset();
		m_prevCameraPos = pos;
		m_prevCameraRot = rot;
	}

	// Tick screenshot before advancing accumulatedTime so the check reads the pre-update value
	if (!m_rasterize)
		m_screenshotManager->Tick(*m_accumulationPass, elapsedTime);

	m_accumulationPass->Update(elapsedTime);

	if (m_placesManager)
		m_placesManager->Tick();
}

void Renderer::Render(double elapsedTime, double totalTime)
{
	if (!m_headless)
	{
		m_editorUI->BeginFrame();
		m_editorUI->EndFrame();
	}

	// In raster mode the active debug view decides how far the VXPG pipeline must
	// run; in raytracing mode the active technique declares its own need.
	const VxpgStage vxpgStage = m_rasterize
		? StageFor(g_rasterizationDebugMode.Get())
		: (m_raytracePass ? m_raytracePass->RequiredVxpgStage() : VxpgStage::None);
	RunVxpgPipelineUpTo(vxpgStage);

	SetViewport();
	SetScissorRect();
	
	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
	
	auto allocator = m_d3d12CommandAllocators[m_frameIndex];
	auto backBuffer = m_d3d12RenderTargets[m_frameIndex];
	
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
		   backBuffer.Get(),
		   D3D12_RESOURCE_STATE_PRESENT,
		   D3D12_RESOURCE_STATE_RENDER_TARGET);

		m_d3d12CommandList->ResourceBarrier(1, &barrier);

		FLOAT clearColor[4] = { 0.3f, 0.6f, 0.9f, 1.0f };

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
			m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			m_frameIndex,
			m_rtvDescriptorSize);
		
		m_d3d12CommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
		m_d3d12CommandList->ClearDepthStencilView(
			m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			D3D12_CLEAR_FLAG_DEPTH,
			1.0f,
			0,
			0,
			nullptr);
		
		m_d3d12CommandList->OMSetRenderTargets(1,
		&rtvHandle,
		true,
		&m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	}
	
	if (m_rasterize)
	{
		ID3D12DescriptorHeap* descriptorHeaps[] = {m_srvCbvUavDescriptorHeap.Get()};

		m_d3d12CommandList->SetPipelineState(m_pipelineStateObject.Get());
		m_d3d12CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
		m_d3d12CommandList->SetGraphicsRootSignature(m_rootSignature.Get());
		
		m_d3d12CommandList->SetGraphicsRootConstantBufferView(3, m_passConstants->GetGpuVirtualAddress());

		if (m_voxelizationPass)
			m_d3d12CommandList->SetGraphicsRootConstantBufferView(4, m_voxelizationPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());

		for (const auto& go : m_scene->GetGameObjects())
		{
			auto gpuAddress = go->m_worldMatrixBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
			m_d3d12CommandList->SetGraphicsRootConstantBufferView(1, gpuAddress);
			
			for (const auto& primitive : go->GetModel()->GetMeshes())
			{
				gpuAddress = primitive->m_material->m_materialBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
				m_d3d12CommandList->SetGraphicsRootConstantBufferView(2, gpuAddress);
				
				auto vertex_view = primitive->GetVertexView();
				auto index_view = primitive->GetIndexView();

				auto vertexBuffer = std::dynamic_pointer_cast<VertexBuffer>(vertex_view.buffer);
				auto indexBuffer = std::dynamic_pointer_cast<IndexBuffer>(index_view.buffer);
	
				m_d3d12CommandList->IASetVertexBuffers(0, 1, &vertexBuffer->GetVertexBufferView());
				m_d3d12CommandList->IASetIndexBuffer(&indexBuffer->GetIndexBufferView());
				m_d3d12CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

				m_d3d12CommandList->SetGraphicsRootDescriptorTable(0, m_srvCbvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
				
				m_d3d12CommandList->DrawIndexedInstanced(index_view.count, 1, index_view.offset, vertex_view.offset, 0);
			}
		}
	}
	else
	{
		{
			ScopedGpuMarker marker(m_d3d12CommandList.Get(), "Raytrace Technique");
			m_raytracePass->Render();
		}

		PostProcessParams postProcessParams;
		postProcessParams.exposure   = g_exposure.Get();
		postProcessParams.contrast   = g_contrast.Get();
		postProcessParams.saturation = g_saturation.Get();
		postProcessParams.lift       = g_lift.Get();

		ScopedGpuMarker postMarker(m_d3d12CommandList.Get(), "Accumulation+PostProcess");
		if (g_accumulationEnabled.Get())
		{
			m_accumulationPass->Render(m_raytracePass->GetOutputResource());
			m_postProcessPass->Render(m_accumulationPass->GetDisplayBuffer(), backBuffer, postProcessParams);
		}
		else
		{
			// Bypass accumulation — normalize raytrace output to COPY_SOURCE for PostProcessPass
			CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
				m_raytracePass->GetOutputResource().Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COPY_SOURCE);
			m_d3d12CommandList->ResourceBarrier(1, &transition);

			m_postProcessPass->Render(m_raytracePass->GetOutputResource(), backBuffer, postProcessParams);

			CD3DX12_RESOURCE_BARRIER transition2 = CD3DX12_RESOURCE_BARRIER::Transition(
				m_raytracePass->GetOutputResource().Get(),
				D3D12_RESOURCE_STATE_COPY_SOURCE,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			m_d3d12CommandList->ResourceBarrier(1, &transition2);
		}

		// Restore main descriptor heap for ImGui (post-process pass may have changed it)
		ID3D12DescriptorHeap* mainHeaps[] = { m_srvCbvUavDescriptorHeap.Get() };
		m_d3d12CommandList->SetDescriptorHeaps(_countof(mainHeaps), mainHeaps);

		// Issue screenshot readback copy if this frame was chosen by Tick()
		if (m_screenshotManager->IsCaptureDue())
			m_screenshotManager->RecordCopy(m_postProcessPass->GetOutputBuffer());
	}

	if (!m_headless)
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_d3d12CommandList.Get());
	
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
			backBuffer.Get(),
			D3D12_RESOURCE_STATE_RENDER_TARGET,
			D3D12_RESOURCE_STATE_PRESENT);

		m_d3d12CommandList->ResourceBarrier(1, &barrier);
		
	}
	
	ID3D12CommandList* const commandLists[] = { m_d3d12CommandList.Get() };
	
	ThrowIfFailed(m_d3d12CommandList->Close());
	
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	UINT presentFlags = m_tearingSupport ? DXGI_PRESENT_ALLOW_TEARING : 0; // TODO: do not check every time
	
	ThrowIfFailed(m_dxgiSwapChain->Present(0, presentFlags));

	FlushCommandQueue();
	ResetCommandList();

	// Map readback buffer and write PNG; GPU is guaranteed done after FlushCommandQueue
	if (m_screenshotManager->IsCaptureDue())
		m_screenshotManager->FinishCapture();
}

void Renderer::CleanUp()
{
	FlushCommandQueue();

	m_editorUI->Shutdown();
	
	m_projectionMatrixConstantBuffer->GetUnderlyingResource()->Unmap(0, nullptr);
}

void Renderer::OnResize()
{
	assert(g_device && "Attempted to resize window without device.");
	assert(m_dxgiSwapChain && "Attempted to resize window without swap chain.");
	
	auto& window = Window::Get();

	FlushCommandQueue();

	for (int i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		m_d3d12RenderTargets[i].Reset();
	}

	int swapChainFlags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	ThrowIfFailed(m_dxgiSwapChain->ResizeBuffers(
		Constants::Graphics::NUM_FRAMES,
		window.GetWidth(),
		window.GetHeight(),
		m_backBufferFormat,
		swapChainFlags));

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		g_device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);
		rtvHandle.Offset(1, m_rtvDescriptorSize);
	}
	
	m_depthStencilBuffer.Reset();

	CreateDepthStencilView();
	SetScissorRect();
	SetViewport();

	m_raytracePass->OnResize();
	m_accumulationPass->OnResize();
	m_postProcessPass->OnResize();

	// Recreate the ShadingPoints G-buffer at the new resolution. Without this it
	// stays at its init size while injection dispatches at the live resolution,
	// writing mismatched rows -> the G-buffer overlay floats / never aligns.
	if (m_vbufferPass)
		m_vbufferPass->OnResize();
	if (m_lightInjectionPass)
		m_lightInjectionPass->OnResize();
	if (m_fingerprintPass)
		m_fingerprintPass->OnResize(window.GetWidth(), window.GetHeight());

	if (m_superpixelBuildPass && m_lightInjectionPass)
	{
		m_superpixelBuildPass->OnResize(window.GetWidth(), window.GetHeight(),
			m_lightInjectionPass->GetShadingPointsTexture().Get());
		WriteSuperpixelUavsToGlobalHeap();
	}

	if (m_clusterVisibilityPass)
	{
		m_clusterVisibilityPass->OnResize(window.GetWidth(), window.GetHeight());
		WriteClusterVisibilityUavsToGlobalHeap();
	}

	if (m_lightTreePass)
		m_lightTreePass->OnResize(window.GetWidth(), window.GetHeight());

	ExecuteCommandsAndReset();
}

void Renderer::OnMouseMove(unsigned long long btnState, int x, int y)
{
	if (ImGui::GetIO().WantCaptureMouse)
	{
		//m_lastMousePosX = x;
		//m_lastMousePosY = y;
		return;
	}

	if((btnState & MK_LBUTTON) != 0)
	{
		// Make each pixel correspond to a quarter of a degree.
		float dx = DirectX::XMConvertToRadians(0.25f*static_cast<float> (x - m_lastMousePosX));
		float dy = DirectX::XMConvertToRadians(0.25f*static_cast<float> (y - m_lastMousePosY));
		// Update angles based on input to orbit camera around box.
		m_theta += dx;
		m_phi += dy;

		m_camera->AddRotationEuler(DirectX::SimpleMath::Vector3(dy, -dx, 0.0f));
		g_cameraRot.Set(m_camera->GetEulerDegrees());

		// Restrict the angle mPhi.
		m_phi = std::clamp(m_phi, 0.01f - DirectX::XM_PIDIV2,  DirectX::XM_PIDIV2 - 0.01f);
	}
	else if((btnState & MK_RBUTTON) != 0)
	{
		// Make each pixel correspond to 0.005 unit in the scene.
		float dx = 0.005f*static_cast<float>(x - m_lastMousePosX);
		float dy = 0.005f*static_cast<float>(y - m_lastMousePosY);
		// Update the camera radius based on input.
		m_radius += dx - dy;
		// Restrict the radius.
		m_radius = std::clamp(m_radius, 3.0f, 15.0f);
	}
	m_lastMousePosX = x;
	m_lastMousePosY = y;
}

void Renderer::OnMouseWheel(int delta)
{
	float scrollFactor = g_cameraScrollFactor.Get();
	float speed = g_cameraSpeed.Get();

	if (delta > 0)
		speed *= scrollFactor;
	else if (delta < 0)
		speed /= scrollFactor;

	g_cameraSpeed.Set(std::clamp(speed, 0.1f, 20.0f));
}

void Renderer::OnKeyDown(unsigned long long btnState) const
{
	const auto state = DirectX::Keyboard::Get().GetState();
	m_keyboardTracker->Update(state);
}

void Renderer::SetupDeviceAndDebug()
{

#ifdef _DEBUG
	ComPtr<ID3D12Debug> debugController;
	D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
	debugController->EnableDebugLayer();

	// DRED: on device-removed, ThrowIfFailed dumps auto-breadcrumbs (which
	// command in which command list hung/faulted) + page-fault allocation info.
	ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSettings;
	if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dredSettings))))
	{
		dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
		dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
	}
#endif
	
	UINT createFactoryFlags = 0;

#ifdef _DEBUG
	createFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

	ThrowIfFailed(::CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory)));

#ifdef _DEBUG
	ComPtr<ID3D12Debug1> debugController1;
	ThrowIfFailed(debugController.As(&debugController1));

	debugController1->SetEnableGPUBasedValidation(ENABLE_GPU_BASED_VALIDATION);
#endif

	if (ComPtr<IDXGIAdapter4> dxgiAdapter = GetHardwareAdapter())
	{
		g_device = GetDeviceForAdapter(dxgiAdapter);
	}
}

void Renderer::CreateCommandQueue()
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};

	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(g_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_d3d12CommandQueue)));
}

void Renderer::CreateCommandAllocators()
{
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_d3d12CommandAllocators[i])));
	}
}

void Renderer::CreateFence()
{
	ThrowIfFailed(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_d3d12Fence)));

	m_fenceValue++;

	m_fenceEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);

	if (m_fenceEvent == nullptr)
	{
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}
}

void Renderer::CreateSwapChain()
{
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = Constants::Graphics::NUM_FRAMES;
	swapChainDesc.Width = Window::Get().GetWidth();
	swapChainDesc.Height = Window::Get().GetHeight();
	swapChainDesc.Format = m_backBufferFormat;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	
	IDXGISwapChain1* swapChain1;
	
	ThrowIfFailed(m_dxgiFactory->CreateSwapChainForHwnd(
		m_d3d12CommandQueue.Get(),
		Window::Get().GetHandle(),
		&swapChainDesc,
		nullptr,
		nullptr,
		&swapChain1));

	ThrowIfFailed(m_dxgiFactory->MakeWindowAssociation(Window::Get().GetHandle(), DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain1->QueryInterface(IID_PPV_ARGS(&m_dxgiSwapChain)));

	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
}

void Renderer::CreateCommandList()
{
	ThrowIfFailed(g_device->CreateCommandList(
		0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_d3d12CommandAllocators[m_frameIndex].Get(),
		nullptr,
		IID_PPV_ARGS(&m_d3d12CommandList)));

	ThrowIfFailed(m_d3d12CommandList->Close());
}

void Renderer::ResetCommandList() const
{
	ThrowIfFailed(m_d3d12CommandAllocators[m_frameIndex]->Reset());
	ThrowIfFailed(m_d3d12CommandList->Reset(m_d3d12CommandAllocators[m_frameIndex].Get(), m_pipelineStateObject.Get()));
}

void Renderer::CreateRTVDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = Constants::Graphics::NUM_FRAMES;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

	ThrowIfFailed(g_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_d3d12RTVDescriptorHeap)));

	m_rtvDescriptorSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void Renderer::CreateRenderTargetViews()
{
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		g_device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);

		rtvHandle.ptr += m_rtvDescriptorSize;
	}
}

void Renderer::CreateDepthStencilView()
{
	D3D12_RESOURCE_DESC desc;
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Alignment = 0;
	desc.Width = Window::Get().GetWidth();
	desc.Height = Window::Get().GetHeight();
	desc.MipLevels = 1;
	desc.Format = m_depthStencilFormat;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
	desc.DepthOrArraySize = 1;

	D3D12_CLEAR_VALUE clearValue;
	clearValue.Format = m_depthStencilFormat;
	clearValue.DepthStencil.Depth = 1.0f;
	clearValue.DepthStencil.Stencil = 0.0f;
	
	CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_DEFAULT);
	
	ThrowIfFailed(g_device->CreateCommittedResource(
		&heapProp,
		D3D12_HEAP_FLAG_NONE,
		&desc,
		D3D12_RESOURCE_STATE_COMMON,
		&clearValue,
		IID_PPV_ARGS(&m_depthStencilBuffer)));

	D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc;
	viewDesc.Flags = D3D12_DSV_FLAG_NONE;
	viewDesc.Format = m_depthStencilFormat;
	viewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	viewDesc.Texture2D.MipSlice = 0;
	
	g_device->CreateDepthStencilView(
		m_depthStencilBuffer.Get(),
		&viewDesc,
		m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	
	m_d3d12CommandList->ResourceBarrier(
		1,
		&CD3DX12_RESOURCE_BARRIER::Transition(
			m_depthStencilBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_DEPTH_WRITE));
}

void Renderer::CreateDSVDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.NumDescriptors = 1;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.NodeMask = 0;

	ThrowIfFailed(g_device->CreateDescriptorHeap(
		&desc,
		IID_PPV_ARGS(&m_d3d12DSVDescriptorHeap)));
}

void Renderer::CreateDescriptorHeaps()
{
	///	|							|						|								|					|					|					|							|
	///	|	SRV IMGUI TEXTURE (1)	|	CBV MATRICES (1)	|	UAV RAYTRACING OUTPUT (1)	|	SRV TLAS (1)	|	VERTEX SRV (1)	|	INDEX SRV (1)	|	TEXTURES (MAX_TEXTURE)	|
	///	|							|						|								|					|					|					|							|
	
	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	desc.NumDescriptors = Constants::Graphics::NUM_BASE_DESCRIPTORS + Constants::Graphics::MAX_TEXTURES + 15; // +1 skybox, +3 voxel, +1 shadingpoints, +2 superpixel index/center, +2 repVPL/vplPosition, +1 vbuffer, +3 cvis gathered/counter/mask, +2 fuzzy weight/index
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	desc.NodeMask = 0;

	ThrowIfFailed(g_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_srvCbvUavDescriptorHeap)));
}

void Renderer::CreateWorldProjCBV()
{
	ComPtr<ID3D12Resource> cbvUav;
	
	g_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(256 * 5),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&cbvUav));

	m_projectionMatrixConstantBuffer = std::make_shared<ConstantBuffer>(g_device, cbvUav);
	m_projectionMatrixConstantBuffer->SetResourceName(L"Camera and World Proj CBV Resource");
	
	ThrowIfFailed(m_projectionMatrixConstantBuffer->GetUnderlyingResource()->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedData)));
	
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
	cbvDesc.BufferLocation = m_projectionMatrixConstantBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
	cbvDesc.SizeInBytes = Align(256ULL * 5, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

	auto descriptorDesc = m_srvCbvUavDescriptorHeap->GetDesc();
	auto increment = g_device->GetDescriptorHandleIncrementSize(descriptorDesc.Type);

	D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
	cbvHandle.ptr = m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + increment;
	
	g_device->CreateConstantBufferView(&cbvDesc, cbvHandle);
}

void Renderer::CreateRasterizationRootSignature()
{
	constexpr int num_params = 5;

	CD3DX12_ROOT_PARAMETER rootParameters[num_params];
	
	D3D12_DESCRIPTOR_RANGE cbvRange;
	cbvRange.BaseShaderRegister = 0;
	cbvRange.NumDescriptors = 1;
	cbvRange.RegisterSpace = 0;
	cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
	cbvRange.OffsetInDescriptorsFromTableStart = 1;

	D3D12_DESCRIPTOR_RANGE rtRange;
	rtRange.BaseShaderRegister = 0;
	rtRange.NumDescriptors = 1;
	rtRange.RegisterSpace = 0;
	rtRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	rtRange.OffsetInDescriptorsFromTableStart = 2;

	D3D12_DESCRIPTOR_RANGE tlasRange;
	tlasRange.BaseShaderRegister = 0;
	tlasRange.NumDescriptors = 1;
	tlasRange.RegisterSpace = 0;
	tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	tlasRange.OffsetInDescriptorsFromTableStart = 3;

	D3D12_DESCRIPTOR_RANGE vertexRange;
	vertexRange.BaseShaderRegister = 1;
	vertexRange.NumDescriptors = 1;
	vertexRange.RegisterSpace = 0;
	vertexRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	vertexRange.OffsetInDescriptorsFromTableStart = 4;
	
	D3D12_DESCRIPTOR_RANGE indexRange;
	indexRange.BaseShaderRegister = 2;
	indexRange.NumDescriptors = 1;
	indexRange.RegisterSpace = 0;
	indexRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	indexRange.OffsetInDescriptorsFromTableStart = 5;
	
	D3D12_DESCRIPTOR_RANGE textureRange;
	textureRange.BaseShaderRegister = 3;
	textureRange.NumDescriptors = Constants::Graphics::MAX_TEXTURES;
	textureRange.RegisterSpace = 0;
	textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	textureRange.OffsetInDescriptorsFromTableStart = 6;

	// u1 = occupancy, u2 = packed irradiance, u3 = vpl count (contiguous heap slots 519..521)
	D3D12_DESCRIPTOR_RANGE voxelOccupancyRange;
	voxelOccupancyRange.BaseShaderRegister = 1;
	voxelOccupancyRange.NumDescriptors = 3;
	voxelOccupancyRange.RegisterSpace = 0;
	voxelOccupancyRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	voxelOccupancyRange.OffsetInDescriptorsFromTableStart = Constants::Graphics::VOXEL_OCCUPANCY_DESCRIPTOR_INDEX;

	// u4 = ShadingPoints G-buffer (debug overlay reads it by screen pixel)
	D3D12_DESCRIPTOR_RANGE shadingPointsRange;
	shadingPointsRange.BaseShaderRegister = 4;
	shadingPointsRange.NumDescriptors = 1;
	shadingPointsRange.RegisterSpace = 0;
	shadingPointsRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	shadingPointsRange.OffsetInDescriptorsFromTableStart = Constants::Graphics::SHADINGPOINTS_DESCRIPTOR_INDEX;

	// u7 = superpixel index, u8 = superpixel representative center (debug views 15/16)
	D3D12_DESCRIPTOR_RANGE superpixelRange;
	superpixelRange.BaseShaderRegister = 7;
	superpixelRange.NumDescriptors = 2;
	superpixelRange.RegisterSpace = 0;
	superpixelRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	superpixelRange.OffsetInDescriptorsFromTableStart = Constants::Graphics::SUPERPIXEL_INDEX_DESCRIPTOR_INDEX;

	D3D12_DESCRIPTOR_RANGE ranges[] = {cbvRange, rtRange, tlasRange, vertexRange, indexRange, textureRange, voxelOccupancyRange, shadingPointsRange, superpixelRange};

	rootParameters[0].InitAsDescriptorTable(_countof(ranges), ranges);
	rootParameters[1].InitAsConstantBufferView(1); // Model index buffer
	rootParameters[2].InitAsConstantBufferView(2); // Material buffer
	rootParameters[3].InitAsConstantBufferView(3); // Pass constants
	rootParameters[4].InitAsConstantBufferView(4); // Voxel grid constants

	auto static_samplers = GetStaticSamplers();
	
	CD3DX12_ROOT_SIGNATURE_DESC desc = {};
	desc.NumParameters = num_params;
	desc.pParameters = rootParameters;
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
	desc.NumStaticSamplers = static_samplers.size();
	desc.pStaticSamplers = static_samplers.data();

	ComPtr<ID3DBlob> serializedRootSignature = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;

	HRESULT hr = D3D12SerializeRootSignature(
		&desc,
		D3D_ROOT_SIGNATURE_VERSION_1,
		&serializedRootSignature,
		&errorBlob);

	if(errorBlob != nullptr)
	{
		OutputDebugStringA(static_cast<char*>(errorBlob->GetBufferPointer()));
	}
	ThrowIfFailed(hr);
	
	ThrowIfFailed(g_device->CreateRootSignature(
		0,
		serializedRootSignature->GetBufferPointer(),
		serializedRootSignature->GetBufferSize(),
		IID_PPV_ARGS(&m_rootSignature)));
}

void Renderer::CreatePipelineState()
{
	auto& rm = ResourceManager::Get();
	
	auto psh = rm.GetOrLoadShader(AssetId("resources/shaders/colorShader.ps.shader"));
	m_pixelShader = rm.shaders.GetResource(psh).bytecode;
	auto vsh = rm.GetOrLoadShader(AssetId("resources/shaders/colorShader.vs.shader"));
	m_vertexShader = rm.shaders.GetResource(vsh).bytecode;
	
	D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
	desc.VS = {static_cast<BYTE*>(m_vertexShader->GetBufferPointer()), m_vertexShader->GetBufferSize()};
	desc.PS = {static_cast<BYTE*>(m_pixelShader->GetBufferPointer()), m_pixelShader->GetBufferSize()};
	desc.InputLayout = {inputLayout, _countof(inputLayout)};
	desc.pRootSignature = m_rootSignature.Get();

	CD3DX12_RASTERIZER_DESC rasterDesc(D3D12_DEFAULT);
	rasterDesc.FrontCounterClockwise = TRUE; // Loaders store canonical CCW winding; CCW is front-facing.
	desc.RasterizerState = rasterDesc;
	desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	
	desc.SampleMask = UINT_MAX;

	desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	
	desc.NumRenderTargets = 1;
	desc.RTVFormats[0] = m_backBufferFormat;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.DSVFormat = m_depthStencilFormat;

	ComPtr<ID3D12PipelineState> pso;
	ThrowIfFailed(g_device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)));
	pso->SetName(L"Default Pipeline State");
	m_pipelineStateObject = pso;
}

void Renderer::SetViewport()
{
	D3D12_VIEWPORT viewport;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Height = Window::Get().GetHeight();
	viewport.Width = Window::Get().GetWidth();
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	m_d3d12CommandList->RSSetViewports(1, &viewport);
}

void Renderer::SetScissorRect()
{
	D3D12_RECT scissorRect;
	scissorRect.left = 0;
	scissorRect.top = 0;
	scissorRect.right = Window::Get().GetWidth();
	scissorRect.bottom = Window::Get().GetHeight();

	m_d3d12CommandList->RSSetScissorRects(1, &scissorRect);
}

void Renderer::FlushCommandQueue()
{
	m_fenceValue++;
	ThrowIfFailed(m_d3d12CommandQueue->Signal(m_d3d12Fence.Get(), m_fenceValue));

	if (m_d3d12Fence->GetCompletedValue() < m_fenceValue)
	{
		ThrowIfFailed(m_d3d12Fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent));
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}

	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
}

bool Renderer::CheckTearingSupport()
{
	BOOL tearingAllowed;

	if (FAILED(m_dxgiFactory->CheckFeatureSupport(
		DXGI_FEATURE_PRESENT_ALLOW_TEARING,
		&tearingAllowed,
		sizeof(tearingAllowed))))
	{
		tearingAllowed = false;
	}
	
	m_tearingSupport = tearingAllowed;
	return tearingAllowed;
}

bool Renderer::CheckRayTracingSupport() const
{
	spdlog::info("Checking ray tracing support...");
	
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
	ThrowIfFailed(g_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
	if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1)
	{
		spdlog::error("Ray tracing not supported!");
		return false;
	}

	// VXPG fingerprint / cvis visibility kernels use inline RayQuery (Tier 1.1,
	// above) plus [WaveSize(32)] ballot packing, which needs shader model 6.6.
	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_6 };
	ThrowIfFailed(g_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)));
	if (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_6)
	{
		spdlog::error("Shader model 6.6 not supported (required by VXPG wave-intrinsic passes)!");
		return false;
	}

	// VXPG light tree: uint64 bitonic sort keys need Int64ShaderOps; the
	// byte-identical uint16 TreeNode layout needs native 16-bit shader ops.
	D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
	ThrowIfFailed(g_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)));
	if (!options1.Int64ShaderOps)
	{
		spdlog::error("Int64 shader ops not supported (required by the VXPG light-tree sort)!");
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4 = {};
	ThrowIfFailed(g_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4)));
	if (!options4.Native16BitShaderOpsSupported)
	{
		spdlog::error("Native 16-bit shader ops not supported (required by the VXPG light-tree nodes)!");
		return false;
	}

	// VXPG guided integrator carries its sampling pdfs in double, faithful to
	// SIByL (ADR 0003 integrator-swap section).
	D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
	ThrowIfFailed(g_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));
	if (!options.DoublePrecisionFloatShaderOps)
	{
		spdlog::error("Double-precision shader ops not supported (required by the VXPG guided integrator pdfs)!");
		return false;
	}

	spdlog::info("Ray tracing supported!");
	return true;
}

void Renderer::CreateTextureSRV(const std::shared_ptr<Texture>& texture)
{
	assert(texture && "Passed texture cannot be null!");
	assert(texture->GetUnderlyingResource() && "Texture resources cannot be null!");
	assert(texture->GetTextureIndex() < Constants::Graphics::MAX_TEXTURES && "Texture index exceeds maximum number of textures supported!");

	spdlog::debug("Setting up texture SRV");

	const auto& resource = texture->GetUnderlyingResource();
	const auto& desc = resource->GetDesc();
	
	D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
	if (desc.DepthOrArraySize == 1)
	{
		srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srv_desc.Texture2D.MostDetailedMip = 0;
		srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
		srv_desc.Texture2D.MipLevels = desc.MipLevels;
	}
	else
	{
		spdlog::error("Texture 2D array functionality is not yet supported!");
		// TODO: In case that texture is a TEXTURE 2D ARRAY
	}
	srv_desc.Format = desc.Format;
	srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	handle.Offset(6 + texture->GetTextureIndex(),  g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));

	g_device->CreateShaderResourceView(resource.Get(), &srv_desc, handle);
}

void Renderer::CreateVertexSRV()
{
	assert(m_scene && "Scene cannot be null when creating vertex SRV");

	auto vertex_buffer = m_scene->GetVertexBuffer();

	assert(vertex_buffer && "Vertex buffer cannot be null when creating vertex SRV!");
	
	D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
	srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srv_desc.Buffer.FirstElement = 0;
	srv_desc.Buffer.NumElements = vertex_buffer->GetBufferSize() / sizeof(uint32_t); // Each element is a single 32bit value -> X Y Z separate, UV separate etc...
	srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
	srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
	srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	handle.Offset(4, g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));

	g_device->CreateShaderResourceView(vertex_buffer->GetUnderlyingResource().Get(), &srv_desc, handle);
}

void Renderer::CreateIndexSRV()
{
	assert(m_scene && "Scene cannot be null when creating index SRV");

	auto index_buffer = m_scene->GetIndexBuffer();

	assert(index_buffer && "Index buffer cannot be null when creating index SRV!");
	
	D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
	srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srv_desc.Buffer.FirstElement = 0;
	srv_desc.Buffer.NumElements = index_buffer->GetIndexCount();
	srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
	srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
	srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	handle.Offset(5, g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));

	g_device->CreateShaderResourceView(index_buffer->GetUnderlyingResource().Get(), &srv_desc, handle);
}

void Renderer::InitializeEditorUI()
{
	m_editorUI = std::make_shared<EditorUI>();
	m_editorUI->Initialize(g_device, m_d3d12CommandQueue, m_srvCbvUavDescriptorHeap);
	m_editorUI->SetCamera(m_camera);
	m_editorUI->SetScene(m_scene);
	m_editorUI->SetAccumulationPass(m_accumulationPass);
	m_editorUI->SetPlacesManager(m_placesManager);
	m_editorUI->SetSkyboxLoadCallback([this](const std::wstring& path) {
		ExecuteCommandsAndReset();
		LoadSkybox(path);
	});
	m_editorUI->SetOnDifferentScenePicked([this](const std::wstring& path) {
		LoadScene(path);
	});
	m_editorUI->SetScreenshotRequestCallback([this](float seconds, std::string modelName, std::string placeName) {
		ArmScreenshot(seconds, modelName, placeName, "", "");
	});
	m_editorUI->SetScreenshotPendingGetter([this]() {
		return m_screenshotManager->IsPending();
	});
	m_editorUI->SetOnDifferentTechniquePicked([this](int index) {
		SetTechniqueByIndex(index);
	});
}

void Renderer::LoadScene(const std::wstring& path)
{
	char pathUtf8[MAX_PATH];
	WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, pathUtf8, sizeof(pathUtf8), nullptr, nullptr);
	spdlog::info("Scene has been changed. Loading: {}", pathUtf8);

	g_textureIndex = 0;
	m_scene = ModelLoading::LoadScene(*this, AssetId(pathUtf8));

	CreateVertexSRV();
	CreateIndexSRV();
	ExecuteCommandsAndReset();

	m_raytracePass->OnSceneChange(m_scene);
	if (m_vbufferPass)
		m_vbufferPass->OnSceneChange(m_scene);
	if (m_lightInjectionPass)
		m_lightInjectionPass->OnSceneChange(m_scene);
	if (m_clusterVisibilityPass)
		m_clusterVisibilityPass->SetScene(m_scene);
	m_editorUI->SetScene(m_scene);
	if (m_placesManager)
		m_placesManager->OnSceneChanged(ExtractModelName(path));
	if (m_voxelizationPass)
		m_voxelizationPass->OnSceneLoaded(*m_scene);
}

void Renderer::SetTechniqueByIndex(int index)
{
	const auto& registry = RaytracePass::GetRegistry();
	if (index < 0 || index >= static_cast<int>(registry.size()))
		return;
	spdlog::info("Switching raytracing technique to: {}", registry[index].name);
	auto newPass = registry[index].create();
	newPass->Initialize(g_device, m_d3d12CommandList, m_scene, m_srvCbvUavDescriptorHeap, m_randomBuffer->GetUnderlyingResource(), m_passConstants);
	m_raytracePass = std::move(newPass);
	m_activeTechniqueIndex = index;
	WireGuidingResources();
}

bool Renderer::SetTechnique(const std::string& name)
{
	const auto& registry = RaytracePass::GetRegistry();
	for (int i = 0; i < static_cast<int>(registry.size()); ++i)
	{
		if (registry[i].name == name)
		{
			SetTechniqueByIndex(i);
			return true;
		}
	}
	return false;
}

std::vector<std::string> Renderer::GetTechniqueNames() const
{
	std::vector<std::string> names;
	for (const auto& entry : RaytracePass::GetRegistry())
		names.push_back(entry.name);
	return names;
}

std::vector<std::string> Renderer::GetPlaceNames() const
{
	std::vector<std::string> names;
	if (m_placesManager)
		for (const Place& place : m_placesManager->GetPlacesForCurrentScene())
			names.push_back(place.name);
	return names;
}

bool Renderer::GoToPlace(const std::string& name)
{
	return m_placesManager && m_placesManager->GoToPlaceByName(name);
}

void Renderer::ArmScreenshot(float seconds, const std::string& model, const std::string& place,
                             const std::string& outDir, const std::string& stem)
{
	m_screenshotManager->SetOutputTarget(outDir, stem);
	ScreenshotMetadata meta = BuildScreenshotMetadata(model, place);
	m_screenshotManager->Arm(*m_accumulationPass, seconds, std::move(meta));
}

bool Renderer::ScreenshotIdle() const
{
	return m_screenshotManager->IsIdle();
}

void Renderer::ApplyRenderConfig(const HeadlessConfig& config)
{
	g_numSamplesPerPixel.Set(static_cast<int32_t>(config.spp));
	g_numBounces.Set(static_cast<int32_t>(config.bounces));
	g_exposure.Set(config.exposure);
	g_contrast.Set(config.contrast);
	g_saturation.Set(config.saturation);
	g_lift.Set(config.lift);
	g_indirectSkyClamp.Set(config.indirectSkyClamp);
	g_skyLighting.Set(config.skyLighting ? 1 : 0);
	g_guidingDebugView.Set(static_cast<GuidingDebugView>(config.guidingDebugView));
	// Headless timed capture integrates over the armed window, so temporal
	// accumulation MUST be on — otherwise every capture is a single frame and
	// --seconds only burns wall-time (the camera is static, so nothing resets it).
	g_accumulationEnabled.Set(1);
}

void Renderer::SetLights(const std::vector<LightData>& lights)
{
	if (!m_scene)
		return;

	m_scene->GetLightDataCPU() = lights;
	m_scene->MarkLightDataDirty();
}

void Renderer::WireGuidingResources()
{
	if (auto guided = std::dynamic_pointer_cast<GuidedPathTracingPass>(m_raytracePass))
		guided->SetGuidingResources(m_voxelizationPass, m_voxelGuidingBuildPass,
			m_fingerprintPass, m_clusterPass, m_lightTreePass);
}

void Renderer::RunVxpgPipelineUpTo(VxpgStage stage)
{
	if (stage == VxpgStage::None || !m_voxelizationPass || !m_scene)
		return;

	// Stage 1: geometry bake (rebakes only when invalidated) + per-frame
	// injection-accumulator clear.
	m_voxelizationPass->SetRuntimeParams(
		g_voxelInjectUseAvg.Get() != 0,
		g_voxelHeatScale.Get());

	// A grid resize destroys grid-sized resources that in-flight frames may
	// still reference — wait for the GPU before recreating anything. Clamp the
	// request the same way the pass does so a persistently out-of-range CVar
	// doesn't flush every frame.
	const uint32_t requestedGridDim =
		std::clamp(static_cast<uint32_t>(g_voxelGridDim.Get()), 4u, 512u);
	if (requestedGridDim != m_voxelizationPass->GetGridDim())
		FlushCommandQueue();

	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG Voxelize/BakeClear");
		m_voxelizationPass->RunFrame(*m_scene, requestedGridDim,
			g_voxelBakeUseCompact.Get() != 0, g_voxelBakeClipping.Get() != 0);
	}
	if (m_voxelizationPass->DidResize())
	{
		WriteVoxelUavsToGlobalHeap();
		// Grid-sized dependents must track the new dim: the inverse index and
		// live bounds are ROOT UAVs (unbounded — undersized would mean GPU
		// memory corruption), the representative texture would silently drop
		// writes.
		if (m_voxelGuidingBuildPass)
			m_voxelGuidingBuildPass->OnVoxelGridResize();
		if (m_lightInjectionPass)
			m_lightInjectionPass->OnVoxelGridResize();
	}

	// Stage 2: shared VBuffer (one jittered primary per pixel, ADR 0004), then
	// light injection reconstructing its first vertex from it (also emits the
	// ShadingPoints G-buffer).
	if (stage >= VxpgStage::Inject && m_vbufferPass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG VBuffer");
		m_vbufferPass->Render();
	}
	if (stage >= VxpgStage::Inject && m_lightInjectionPass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG LightInjection");
		m_lightInjectionPass->Render();
	}

	// Stage 3: build the guiding distribution from the injected voxels
	// (reload baked bounds -> compact -> CDF).
	if (stage >= VxpgStage::GuidingBuild && m_voxelGuidingBuildPass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG GuidingBuild");
		m_voxelGuidingBuildPass->Run(
			m_lightInjectionPass ? m_lightInjectionPass->GetVoxelRepresentativeTexture().Get() : nullptr);
	}

	// Stage 4: fingerprint every lit voxel (128 stratified screen representatives
	// -> per-voxel visibility mask via inline shadow rays).
	if (stage >= VxpgStage::Fingerprint && m_fingerprintPass && m_scene)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG Fingerprint");
		auto tlas = m_scene->GetAccelerationStructures()->GetTopLevelAS().p_result;
		m_fingerprintPass->Run(tlas ? tlas->GetGPUVirtualAddress() : 0,
			m_passConstants->data.frameIndex);
	}

	// Stage 5: k-means++ cluster the fingerprinted voxels into 32 supervoxels.
	if (stage >= VxpgStage::Cluster && m_clusterPass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG Cluster");
		m_clusterPass->Run(m_passConstants->data.frameIndex);
	}

	// Stage 6: SLIC superpixel clustering over the ShadingPoints G-buffer.
	if (stage >= VxpgStage::Superpixel && m_superpixelBuildPass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG Superpixel");
		m_superpixelBuildPass->Run(
			m_lightInjectionPass ? m_lightInjectionPass->GetShadingPointsTexture().Get() : nullptr,
			g_superpixelWeight.Get(), g_superpixelPosNormalizer.Get());
	}

	// Stage 7: per-superpixel x per-cluster soft visibility (cvis).
	if (stage >= VxpgStage::ClusterVisibility && m_clusterVisibilityPass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG ClusterVisibility");
		m_clusterVisibilityPass->Run(m_passConstants->data.frameIndex);
	}

	// Stage 8: bottom light tree (Karras LBVH over lit voxels: encode -> sort ->
	// initial -> internal -> merge).
	if (stage >= VxpgStage::LightTree && m_lightTreePass)
	{
		ScopedGpuMarker marker(m_d3d12CommandList.Get(), "VXPG LightTree");
		m_lightTreePass->Run();
	}
}

void Renderer::WriteVoxelUavsToGlobalHeap()
{
	if (!m_voxelizationPass)
		return;

	auto descSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE heapStart = m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_CPU_DESCRIPTOR_HANDLE slot = heapStart;
	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::VOXEL_OCCUPANCY_DESCRIPTOR_INDEX) * descSize;
	m_voxelizationPass->WriteOccupancyUavTo(slot);

	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::VOXEL_IRRADIANCE_DESCRIPTOR_INDEX) * descSize;
	m_voxelizationPass->WriteIrradianceUavTo(slot);

	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::VOXEL_VPL_COUNT_DESCRIPTOR_INDEX) * descSize;
	m_voxelizationPass->WriteVplCountUavTo(slot);
}

void Renderer::WriteSuperpixelUavsToGlobalHeap()
{
	if (!m_superpixelBuildPass)
		return;

	auto descSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE heapStart = m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_CPU_DESCRIPTOR_HANDLE slot = heapStart;
	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::SUPERPIXEL_INDEX_DESCRIPTOR_INDEX) * descSize;
	m_superpixelBuildPass->WriteIndexUavTo(slot);

	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::SUPERPIXEL_CENTER_DESCRIPTOR_INDEX) * descSize;
	m_superpixelBuildPass->WriteCenterUavTo(slot);

	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::FUZZY_WEIGHT_DESCRIPTOR_INDEX) * descSize;
	m_superpixelBuildPass->WriteFuzzyWeightUavTo(slot);

	slot.ptr = heapStart.ptr + static_cast<SIZE_T>(Constants::Graphics::FUZZY_INDEX_DESCRIPTOR_INDEX) * descSize;
	m_superpixelBuildPass->WriteFuzzyIndexUavTo(slot);
}

void Renderer::WriteClusterVisibilityUavsToGlobalHeap()
{
	if (!m_clusterVisibilityPass || !m_superpixelBuildPass)
		return;

	auto descSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE heapStart = m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	auto writeUav = [&](int index, ID3D12Resource* res, DXGI_FORMAT fmt)
	{
		if (!res) return;
		D3D12_CPU_DESCRIPTOR_HANDLE slot = heapStart;
		slot.ptr = heapStart.ptr + static_cast<SIZE_T>(index) * descSize;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
		uav.Format        = fmt;
		uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		g_device->CreateUnorderedAccessView(res, nullptr, &uav, slot);
	};

	writeUav(Constants::Graphics::SPIXEL_GATHERED_DESCRIPTOR_INDEX,
		m_superpixelBuildPass->GetGatheredResource(), DXGI_FORMAT_R32G32_SINT);
	writeUav(Constants::Graphics::SPIXEL_COUNTER_DESCRIPTOR_INDEX,
		m_superpixelBuildPass->GetCounterResource(), DXGI_FORMAT_R32_UINT);
	writeUav(Constants::Graphics::CLUSTER_VISIBILITY_MASK_DESCRIPTOR_INDEX,
		m_clusterVisibilityPass->GetMaskResource(), DXGI_FORMAT_R32_UINT);
}

void Renderer::LoadSkybox(const std::wstring& path)
{
	ComPtr<ID3D12Resource> textureResource;
	std::unique_ptr<uint8_t[]> ddsData;
	std::vector<D3D12_SUBRESOURCE_DATA> subresources;

	HRESULT hr = DirectX::LoadDDSTextureFromFile(
		g_device.Get(), path.c_str(),
		&textureResource, ddsData, subresources);

	if (FAILED(hr))
	{
		spdlog::error("Failed to load skybox DDS: {}", std::system_category().message(hr));
		return;
	}

	// Upload subresources
	const UINT64 uploadSize = GetRequiredIntermediateSize(textureResource.Get(), 0, static_cast<UINT>(subresources.size()));

	ComPtr<ID3D12Resource> uploadBuffer;
	auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
	ThrowIfFailed(g_device->CreateCommittedResource(
		&heapProps, D3D12_HEAP_FLAG_NONE,
		&bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr, IID_PPV_ARGS(&uploadBuffer)));

	{
		auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(textureResource.Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		m_d3d12CommandList->ResourceBarrier(1, &barrier);
	}

	UpdateSubresources(m_d3d12CommandList.Get(), textureResource.Get(), uploadBuffer.Get(),
		0, 0, static_cast<UINT>(subresources.size()), subresources.data());

	{
		auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(textureResource.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		m_d3d12CommandList->ResourceBarrier(1, &barrier);
	}

	// Create SRV at SKYBOX_DESCRIPTOR_INDEX
	auto desc = textureResource->GetDesc();
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = desc.Format;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = desc.MipLevels;

	UINT descriptorSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
		m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
		Constants::Graphics::SKYBOX_DESCRIPTOR_INDEX, descriptorSize);

	g_device->CreateShaderResourceView(textureResource.Get(), &srvDesc, srvHandle);

	m_skyboxResource = textureResource;

	ExecuteCommandsAndReset();
	spdlog::info("Skybox loaded successfully.");
}

void Renderer::OnShaderReload()
{
	spdlog::info("Reloading shaders...");
	ResourceManager::Get().RecompileAllShaders();
	
	ThrowIfFailed(m_d3d12CommandList->Close());
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

	FlushCommandQueue();
	
	spdlog::info("Creating pipeline state for new shaders...");
	CreatePipelineState();
	m_raytracePass->OnShaderReload();
	if (m_vbufferPass)
		m_vbufferPass->OnShaderReload();
	if (m_lightInjectionPass)
		m_lightInjectionPass->OnShaderReload();

	FlushCommandQueue();
	ResetCommandList();
	spdlog::info("Shaders reloaded successfully.");
}

std::array<const CD3DX12_STATIC_SAMPLER_DESC, Constants::Graphics::STATIC_SAMPLERS_COUNT> Renderer::GetStaticSamplers()
{
	// Apps usually only need a handful of samplers. So just define them
	// all up front and keep them available as part of the root signature.

	const CD3DX12_STATIC_SAMPLER_DESC pointWrap(
		0, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
		1, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC linearWrap(
		2, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC linearClamp(
		3, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(
		4, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressW
		0.0f, // mipLODBias
		8); // maxAnisotropy
	const CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(
		5, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressW
		0.0f, // mipLODBias
		8); // maxAnisotropy
	;

	return { pointWrap, pointClamp, linearWrap, linearClamp, anisotropicWrap, anisotropicClamp };
}

void Renderer::ToggleRasterization()
{
	m_rasterize = !m_rasterize;
}

std::pair<std::shared_ptr<VertexBuffer>, std::shared_ptr<IndexBuffer>> Renderer::CreateSceneResources(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
	ComPtr<ID3D12Resource> vertex_upload_buffer;
	ComPtr<ID3D12Resource> index_upload_buffer;

	auto cpuVertex = static_cast<BYTE*>(malloc(vertices.size() * sizeof(Vertex)));
	auto cpuIndex = static_cast<BYTE*>(malloc(indices.size() * sizeof(uint32_t)));
	memcpy(cpuVertex, vertices.data(), vertices.size() * sizeof(Vertex));
	memcpy(cpuIndex, indices.data(), indices.size() * sizeof(uint32_t));
	
	auto vertex_buffer_resource = RenderingUtils::CreateDefaultBuffer(g_device.Get(), m_d3d12CommandList.Get(), cpuVertex, vertices.size() * sizeof(Vertex), vertex_upload_buffer);
	auto index_buffer_resource = RenderingUtils::CreateDefaultBuffer(g_device.Get(), m_d3d12CommandList.Get(), cpuIndex, indices.size() * sizeof(uint32_t), index_upload_buffer);
	
	// Need to be closed and executed to create buffers before the upload buffers go out of scope.
	ExecuteCommandsAndReset();

	auto vertex_buffer = std::make_shared<VertexBuffer>(g_device, vertex_buffer_resource, static_cast<UINT>(vertices.size()), sizeof(Vertex));
	auto index_buffer = std::make_shared<IndexBuffer>(g_device, index_buffer_resource, static_cast<UINT>(indices.size()), DXGI_FORMAT_R32_UINT);
	
	AssertFreeClear(&cpuVertex);
	AssertFreeClear(&cpuIndex);
	
	return std::make_pair(vertex_buffer, index_buffer);
}

std::shared_ptr<Texture> Renderer::CreateTextureFromGLTF(const tinygltf::Image& image)
{
	ComPtr<ID3D12Resource> upload_buffer;

	auto texture_resource = RenderingUtils::CreateDefaultTexture(g_device.Get(), m_d3d12CommandList.Get(), image, upload_buffer);

	std::shared_ptr<Texture> texture = std::make_shared<Texture>(g_device, texture_resource);
	CreateTextureSRV(texture);

	m_d3d12CommandList->Close();
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	FlushCommandQueue();
	ResetCommandList();

	m_textures.push_back(texture);
	
	return texture;
}

ComPtr<IDXGIAdapter4> Renderer::GetHardwareAdapter(bool useWarp)
{
	ComPtr<IDXGIAdapter1> adapter1;
	ComPtr<IDXGIAdapter4> adapter;

	if (useWarp)
	{
		ThrowIfFailed(m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter1)));
		ThrowIfFailed(adapter1.As(&adapter));
		return adapter;
	}

	size_t maxDedicatedVideoMemory = 0;

	for (UINT i = 0; m_dxgiFactory->EnumAdapters1(i, &adapter1) != DXGI_ERROR_NOT_FOUND; ++i){
		DXGI_ADAPTER_DESC1 desc;
		adapter1->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			continue;

		if (SUCCEEDED(D3D12CreateDevice(adapter1.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
		{
			if (desc.DedicatedVideoMemory > maxDedicatedVideoMemory)
			{
				maxDedicatedVideoMemory = desc.DedicatedVideoMemory;
				ThrowIfFailed(adapter1.As(&adapter));
			}
		}
	}

	return adapter;
}

ComPtr<ID3D12Device5> Renderer::GetDeviceForAdapter(ComPtr<IDXGIAdapter1> adapter)
{
	ComPtr<ID3D12Device5> device;
	ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

#ifdef _DEBUG
	ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_infoQueue)));
	// Only break into the debugger on genuine memory corruption; errors and
	// warnings are logged (below) so headless runs surface them instead of
	// aborting on a breakpoint with no debugger attached.
	ThrowIfFailed(m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true));
	m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, false);
	m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, false);

	// Mirror every debug-layer message into spdlog (console + headless log).
	ComPtr<ID3D12InfoQueue1> infoQueue1;
	if (SUCCEEDED(m_infoQueue.As(&infoQueue1)))
	{
		infoQueue1->RegisterMessageCallback(&D3D12DebugMessageCallback,
			D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &m_debugMessageCallbackCookie);
	}
	else
	{
		spdlog::warn("ID3D12InfoQueue1 unavailable; D3D12 messages will only reach the debugger output.");
	}
#endif


	return device;
}

std::shared_ptr<GameObject> Renderer::InstantiateGameObject()
{
	ComPtr<ID3D12Resource> buffer;
	
	g_device->CreateCommittedResource(
	&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
	D3D12_HEAP_FLAG_NONE,
	&CD3DX12_RESOURCE_DESC::Buffer(256),
	D3D12_RESOURCE_STATE_GENERIC_READ,
	nullptr,
	IID_PPV_ARGS(&buffer));

	auto constantBuffer = std::make_shared<ConstantBuffer>(g_device, buffer);
	constantBuffer->SetResourceName(L"Model CBV Resource");

	auto game_object = std::make_shared<GameObject>();
	game_object->m_worldMatrixBuffer = constantBuffer;

	auto matrix = DirectX::XMMatrixIdentity();
	DirectX::XMFLOAT4X4 modelWorldMatrix;
	DirectX::XMStoreFloat4x4(&modelWorldMatrix, matrix);
	game_object->UpdateWorldMatrix(modelWorldMatrix);

	return game_object;
}

void Renderer::ExecuteCommandsAndReset()
{
	ThrowIfFailed(m_d3d12CommandList->Close());
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	FlushCommandQueue();
	// WHY is resetting the allocator impossible if:
	// 1. commands are closed
	// 2. commands are executed
	// 3. command queue is flushed
	// D3D12 ERROR: ID3D12CommandAllocator::Reset: The command allocator cannot be reset because a command list is currently being recorded with the allocator. [ EXECUTION ERROR #543: COMMAND_ALLOCATOR_CANNOT_RESET]
	// Is it possible that in the meantime something gets recorded to the command list?
	// ---
	// We have 3 allocators for each of the triple buffered frames
	// When we reset, we reset only the current frame's allocator
	// But maybe we haven't yet presented the frame, and we need to??? is that it?

	// After all it seems that the FlushGPU method was not functioning correctly, I never quite researched why that was the case. But it seems that there was an issue with the fence value.
	// There was a unique fence value for each allocator (frame) and somehow it was not being updated properly. TODO: Check out why was that for the next iteration of the engine.
	ResetCommandList();
}

ScreenshotMetadata Renderer::BuildScreenshotMetadata(const std::string& modelName, const std::string& placeName) const
{
    ScreenshotMetadata m;
    m.modelName = modelName;
    m.placeName = placeName;

    if (m_camera)
    {
        m.cameraPosition = m_camera->GetPosition();
        m.cameraRotation = m_camera->GetRotation();
        m.cameraFov      = m_camera->GetFovYRadians();
    }

    const auto& registry = RaytracePass::GetRegistry();
    if (m_activeTechniqueIndex >= 0 && m_activeTechniqueIndex < static_cast<int>(registry.size()))
        m.techniqueName = registry[m_activeTechniqueIndex].name;

    m.postProcessEnabled = true;
    m.exposure   = g_exposure.Get();
    m.contrast   = g_contrast.Get();
    m.saturation = g_saturation.Get();
    m.lift       = g_lift.Get();

    m.samplesPerPixel = static_cast<uint32_t>(g_numSamplesPerPixel.Get());
    m.bounces         = static_cast<uint32_t>(g_numBounces.Get());

    return m;
}
