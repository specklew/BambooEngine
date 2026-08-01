#include "pch.h"

#include "Renderer.h"

#include <algorithm>
#include <crtdbg.h>

#include "Camera.h"
#include "DDSTextureLoader/DDSTextureLoader12.h"
#include "EditorUI.h"
#include "FrameAccumulationPass.h"
#include "StatesManager.h"
#include "PostProcessPass.h"
#include "Utils/CVars.h"
#include "Utils/GpuMarker.h"
#include "Utils/Utils.h"
#include "SceneResources/GameObject.h"
#include "imgui.h"
#include "backends/imgui_impl_dx12.h"

#include "CommandContext.h"
#include "InputElements.h"
#include "ShaderProgram.h"
#include "Resources/ResourceStateTracker.h"
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

using namespace Microsoft::WRL;

namespace
{
    std::string ExtractModelName(const std::filesystem::path& p)
    {
        // Tungsten scenes are all named scene.json — key by the containing folder
        // so they don't collide on "scene" and each gets its own places entry.
        if (ToLowerAscii(p.extension().string()) == ".json")
            return ToLowerAscii(p.parent_path().filename().string());
        return ToLowerAscii(p.stem().string());
    }
    std::string ExtractModelName(const std::string& path)
    {
        return ExtractModelName(std::filesystem::path(path));
    }
    std::string ExtractModelName(const std::wstring& path)
    {
        return ExtractModelName(std::filesystem::path(path));
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
// Opt-in stripped raygen (debug-view code compiled out). Measured SLOWER on
// the current AMD RDNA driver (see the variant-sync block in Update); kept for
// A/Bs on other vendors/drivers where the dead-code footprint may win.
static AutoCVarInt g_raygenCleanVariant("renderer.raygenCleanVariant", "1 = compile raygen without debug-view code (measured slower on RDNA)", 0, CVarFlags::EditCheckbox);
static AutoCVarInt g_numSamplesPerPixel("renderer.samplesPerPixel", "Number of samples per pixel", 1, CVarFlags::EditDrag, 1, 64);
static AutoCVarInt g_numBounces("renderer.numBounces", "Number of bounces", 1, CVarFlags::EditDrag, 0, 7);
static AutoCVarInt   g_accumulationEnabled("renderer.accumulation.enabled","Enable temporal frame accumulation when camera is still", 0, CVarFlags::EditCheckbox);
// One-shot: set to 1 to log the next frame's graph (nodes, declarations, the
// barriers they synthesized), then it clears itself.
static AutoCVarInt   g_dumpRenderGraph("rdg.dump", "Log the next frame's render graph and its synthesized barriers", 0, CVarFlags::EditCheckbox);
// Off by default: two timestamps per node plus a resolve is real per-frame cost,
// and benchmark runs must measure the renderer, not the instrumentation.
static AutoCVarInt   g_renderGraphTimings("rdg.timings", "Measure each render-graph node on the GPU (ImGui: Render Graph)", 0, CVarFlags::EditCheckbox);
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
// ADR 0009: 1 = the guided GI's BSDF MIS subtree writes the VPL fitting data
// (last-frame reuse, supplemental 2's own pattern) and the injection pass
// shrinks to a ShadingPoints writer; 0 = SIByL-shipped dedicated injection trace.
static AutoCVarInt g_injectionReuseGi("vxpg.injection.reuseGiSamples",
	"Fit the guide from last frame's GI BSDF samples instead of a dedicated injection trace",
	1, CVarFlags::EditCheckbox);
// Default = power: the ported integrator (vxguiding-gi strategy 5) hardcodes
// power-heuristic squaring; balance stays available as a variance experiment.
static AutoCVarInt   g_guidingPowerMis("guiding.powerMis", "MIS heuristic: 0 = balance, 1 = power", 1, CVarFlags::EditCheckbox);
// Bottom light-tree branch weighting (guidedPathTracing.hlsl FirstChildProb).
// 0 = intensity-only (telescoping reverse pdf, the shipped SIByL strategy-5
// default); 1 = SIByL SLC geometry bound + avg-minmax distance (the paper's
// distanceType==2); 2 = same but the CHEAP GeomTermBoundApproximate (drops the
// tangent frame + two 8-corner passes, ~1/5 the per-node geometry cost). Modes
// 1/2 make the within-cluster voxel pick account for solid angle + orientation,
// at the cost of a non-telescoping leaf->root reverse pdf walk per BSDF-MIS
// query. Rides guidingFlags bits 5-6.
static AutoCVarInt   g_guidingTreeWeightMode("vxpg.tree.weightMode", "Bottom light-tree weighting: 0 = intensity-only, 1 = geometry exact + dist (paper), 2 = geometry approx + dist (cheap)", 0, CVarFlags::EditDrag, 0, 2);
// Second-bounce guiding (SIByL strategy-6 `second=true`, guidedPathTracing.hlsl
// ShadeSecondVertex). MIS-guides the second path vertex via the global voxel
// irradiance guide, turning the BSDF branch into a 2-bounce guided path.
// Meaningful only at bounces >= 2; default off. Rides guidingFlags bit 7.
static AutoCVarInt   g_guidingSecondBounce("vxpg.secondBounce", "Also MIS-guide the second path vertex (SIByL second=true); needs bounces >= 2", 0, CVarFlags::EditCheckbox);
// One-sample MIS at the first vertex (ADR 0015, deviation from SIByL's
// two-sample MIS): a fair coin picks EITHER the BSDF or the guide strategy
// per sample and only that branch traces — halves the GI raygen's trace work
// at higher per-sample variance. Debug views keep the two-sample estimator.
// Rides guidingFlags bit 8.
static AutoCVarInt   g_guidingOneSampleMis("vxpg.oneSampleMis", "One-sample MIS: trace one stochastically-picked strategy per sample instead of both", 0, CVarFlags::EditCheckbox);
// Adaptive per-tile selection probability for one-sample MIS (ADR 0015):
// learned from the previous frame's per-strategy contribution shares —
// where one strategy dominates, one-sample loses almost no variance there.
// Rides guidingFlags bit 9; 0 = fixed fair coin.
static AutoCVarInt   g_guidingOneSampleAdaptiveQ("vxpg.oneSample.adaptiveQ", "Adaptive per-tile strategy-selection probability for one-sample MIS; 0 = fixed 0.5", 1, CVarFlags::EditCheckbox);
static AutoCVarFloat g_indirectSkyClamp("pathtracing.indirectSkyClamp", "Clamp indirect-bounce skybox radiance to suppress HDR-sun fireflies for benchmark convergence. 0 = disabled (unbiased)", 0.0f, CVarFlags::EditDrag, 0.0f, 1000.0f);
static AutoCVarInt   g_skyLighting("pathtracing.skyLighting", "Skybox radiance lights surfaces via indirect rays; 0 = sky is background-only (benchmark isolation: the VXPG guide only targets direct-lit surfaces)", 1, CVarFlags::EditCheckbox);
static AutoCVarEnum  g_guidingDebugView("guiding.debugView", "Guided PT debug visualization", GuidingDebugView::None,
                                        CVarFlags::None, FormatDebugViewDocs<GuidingDebugView>(kGuidingDebugViewDocs));

void Renderer::Initialize()
{
	spdlog::info("Initializing renderer...");

	m_camera = std::make_shared<Camera>();
	m_keyboardTracker = std::make_shared<DirectX::Keyboard::KeyboardStateTracker>();

	m_statesManager = std::make_shared<StatesManager>();
	m_statesManager->Load();
	m_statesManager->SetCamera(*m_camera);
	m_statesManager->SetLightsAccessors(
		[this]() -> std::vector<LightData> {
			return m_scene ? m_scene->GetLightDataCPU() : std::vector<LightData>{};
		},
		[this](const std::vector<LightData>& lights) {
			SetLights(lights);   // assigns scene light list + MarkLightDataDirty
		});
	
#ifdef _DEBUG
	// Route CRT assertion failures to stderr instead of a modal dialog, so
	// asserts appear in the console / headless log rather than hanging the run.
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

	m_graphicsDevice = std::make_unique<GraphicsDevice>();
	m_graphicsDevice->Initialize(g_enableDebugLayer);
	g_device = m_graphicsDevice->GetDevice();
	m_graphicsDevice->CheckTearingSupport();

	if (!m_graphicsDevice->CheckRayTracingSupport()) 	throw std::runtime_error("Raytracing is not supported on this device.");;

	m_graphicsDevice->CreateCommandQueue();
	m_graphicsDevice->CreateCommandAllocators();
	m_graphicsDevice->CreateFence();
	m_graphicsDevice->CreateSwapChain(Window::Get().GetHandle(), Window::Get().GetWidth(),
		Window::Get().GetHeight(), m_backBufferFormat);

	CreateCommandList();
	ResetCommandList();

	m_renderGraph.InitializeTimers(g_device.Get(), m_graphicsDevice->GetCommandQueue().Get());

	CreateRTVDescriptorHeap();
	CreateRenderTargetViews();

	CreateDSVDescriptorHeap();
	CreateDepthStencilView();

	// Execute depth stencil creation commands
	ExecuteCommandsAndReset();
	// Finish execution - reset command list for next setup commands

	GlobalDescriptorHeap::Get().Initialize(g_device.Get());
	CreateWorldProjCBV();

	m_scene = ModelLoading::LoadScene(*this, AssetId("resources/models/abeautifulgame.glb"));
	m_statesManager->OnSceneChanged(ExtractModelName(std::string("resources/models/abeautifulgame.glb")));

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
	m_raytracePass->Initialize(g_device, m_d3d12CommandList, m_scene, m_randomBuffer->GetUnderlyingResource(), m_passConstants);

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
	m_vbufferPass->Initialize(g_device, m_d3d12CommandList, m_scene, m_randomBuffer->GetUnderlyingResource(), m_passConstants);

	m_lightInjectionPass = std::make_shared<LightInjectionPass>();
	m_lightInjectionPass->SetVoxelizationPass(m_voxelizationPass);
	m_lightInjectionPass->Initialize(g_device, m_d3d12CommandList, m_scene, m_randomBuffer->GetUnderlyingResource(), m_passConstants);

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
	m_clusterVisibilityPass->Initialize(g_device, m_d3d12CommandList,
		m_voxelizationPass, m_voxelGuidingBuildPass, m_clusterPass, m_superpixelBuildPass);
	m_clusterVisibilityPass->SetScene(m_scene);
	m_clusterVisibilityPass->OnResize(Window::Get().GetWidth(), Window::Get().GetHeight());
	WriteClusterVisibilityUavsToGlobalHeap();

	m_lightTreePass = std::make_shared<VxpgLightTreePass>();
	m_lightTreePass->Initialize(g_device, m_d3d12CommandList,
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
	// Applying every frame would stomp camera state set via other paths (e.g. StatesManager::GoTo).
	// Headless drives the camera solely through GoToState, so this sync is skipped there.
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

	// Debug-view shader variant sync. Default keeps the view code compiled in:
	// measured FASTER on the current AMD RDNA driver (interleaved A/B veach-ajar
	// Deep Light 3s: views-in PT 3884/3874/3823, VXPG 833/826/832 vs stripped
	// PT 3653/3656/3588/3556, VXPG 762/764/763/760 — stripping costs 6-8%,
	// suspected driver wave-size heuristic flip, cf. ADR 0011 wave64 -5%).
	// renderer.raygenCleanVariant=1 opts into the stripped raygen for vendor
	// A/Bs; an active debug-view CVar always forces the view code in. A
	// transition swaps the sidecar (GetTechniqueDesc) and goes through the full
	// OnShaderReload path (flush + pipeline/SBT rebuild), same as F2 — rare.
	if (!m_rasterize)
	{
		const bool viewActive = m_raytracePass->UsesVoxelGuiding()
			? g_guidingDebugView.Get() != GuidingDebugView::None
			: g_raytraceDebugMode.Get() != RaytraceDebugMode::None;
		const bool wantsDebugViews = viewActive || g_raygenCleanVariant.Get() == 0;
		bool needsReload = m_raytracePass->SetDebugViewsCompiled(wantsDebugViews);
		needsReload |= m_raytracePass->SetOneSampleMisCompiled(g_guidingOneSampleMis.Get() != 0);
		if (needsReload)
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

	// Must run before the PassConstants fill below: it reads the pool's fresh
	// count/total-power off m_scene, and the analytic tail rebuild changes them
	// when light data is dirty (headless overrides, EditorUI edits).
	if (m_scene->IsLightDataDirty())
	{
		m_scene->SetLightDataBuffer(CreateStructuredBuffer(m_scene->GetLightDataCPU()));
		m_scene->RebuildLightPoolAnalyticTail(*this);
		m_scene->ClearLightDataDirty();
	}

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
		((static_cast<uint32_t>(g_guidingDebugView.Get()) & 15u) << 1) |
		((static_cast<uint32_t>(g_guidingTreeWeightMode.Get()) & 3u) << 5) |
		((g_guidingSecondBounce.Get() != 0 ? 1u : 0u) << 7) |
		((g_guidingOneSampleMis.Get() != 0 ? 1u : 0u) << 8) |
		((g_guidingOneSampleAdaptiveQ.Get() != 0 ? 1u : 0u) << 9);
	static_assert(static_cast<int>(GuidingDebugView::SymmetricBsdfBaseline) <= 15, "GuidingDebugView must fit in 4 bits of guidingFlags");
	const auto& camPos = m_camera->GetPosition();
	m_passConstants->data.cameraWorldPos = { camPos.x, camPos.y, camPos.z };
	m_passConstants->data.numLights = m_scene->GetLightDataBuffer()->GetElementsCount();
	// Per-pixel jitter is derived in-shader from (pixel, frameIndex); the CB
	// just carries the on/off switch.
	m_passConstants->data.vbufferJitterEnabled = (g_vbufferJitter.Get() != 0) ? 1u : 0u;
	m_passConstants->data.indirectSkyClamp = g_indirectSkyClamp.Get();
	m_passConstants->data.skyLightingEnabled = (g_skyLighting.Get() != 0) ? 1u : 0u;
	m_passConstants->data.lightPoolCount = m_scene->GetLightPoolCount();
	m_passConstants->data.lightPoolTotalPower = m_scene->GetLightPoolTotalPower();
	m_passConstants->Map();

	// Camera change detection for accumulation reset. Skipped in headless: the
	// camera only changes between captures (GoToState), and ArmScreenshot owns the
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

	if (m_statesManager)
		m_statesManager->Tick();
}

void Renderer::Render(double elapsedTime, double totalTime)
{
	if (!m_headless)
	{
		m_editorUI->BeginFrame();
		m_editorUI->EndFrame();
	}

	// The frame's graph starts here: every VXPG node is added, then (in raytracing
	// mode) the integrator and present chain are appended below. Which VXPG stages
	// actually run is decided by culling, from what the frame's consumer declares.
	m_renderGraph.Reset();
	m_renderGraph.SetBarrierLogging(g_dumpRenderGraph.Get() != 0);
	m_renderGraph.SetTimingEnabled(g_renderGraphTimings.Get() != 0);
	BuildVxpgGraph();

	SetViewport();
	SetScissorRect();
	
	m_graphicsDevice->RefreshFrameIndex();
	const UINT frameIndex = m_graphicsDevice->GetFrameIndex();

	Texture& backBuffer = *m_backBufferTextures[frameIndex];

	{
		backBuffer.TransitionChecked(m_d3d12CommandList.Get(),
			D3D12_RESOURCE_STATE_PRESENT,
			D3D12_RESOURCE_STATE_RENDER_TARGET);

		FLOAT clearColor[4] = { 0.3f, 0.6f, 0.9f, 1.0f };

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
			m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			frameIndex,
			m_rtvDescriptorSize);

		// Raytracing overwrites the whole back buffer with the present copy and
		// draws no geometry, so both clears are dead bandwidth there (measured
		// 2026-08-01, ABeautifulGame 3 s Debug: PT 3151 -> 3276 frames).
		if (m_rasterize)
		{
			CommandContext::Get().GetCommandList()->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
			CommandContext::Get().GetCommandList()->ClearDepthStencilView(
				m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
				D3D12_CLEAR_FLAG_DEPTH,
				1.0f,
				0,
				0,
				nullptr);
		}

		m_d3d12CommandList->OMSetRenderTargets(1,
		&rtvHandle,
		true,
		&m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	}
	
	if (m_rasterize)
	{
		// Raster draws are not graph nodes yet (phase 5). This sink stands in for
		// them so culling keeps exactly the VXPG stages the active debug view reads.
		m_renderGraph.AddPass("Raster Debug View",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.NeverCull();
				DeclareRasterDebugViewReads(pass);
			},
			nullptr);

		m_renderGraph.Compile();
		m_renderGraph.Execute(CommandContext::Get());

		ID3D12DescriptorHeap* descriptorHeaps[] = {GlobalDescriptorHeap::Get().GetHeap()};

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

				m_d3d12CommandList->SetGraphicsRootDescriptorTable(0, GlobalDescriptorHeap::Get().GpuStart());
				
				CommandContext::Get().DrawIndexedInstanced(index_view.count, 1, index_view.offset, vertex_view.offset, 0);
			}
		}
	}
	else
	{
		PostProcessParams postProcessParams;
		postProcessParams.exposure   = g_exposure.Get();
		postProcessParams.contrast   = g_contrast.Get();
		postProcessParams.saturation = g_saturation.Get();
		postProcessParams.lift       = g_lift.Get();

		// ADR 0017 step A: the raytrace -> accumulate -> tonemap -> copy chain
		// declares what it touches and the graph places the barriers. The passes
		// themselves no longer carry any.
		Texture& raytraceOutput = *m_raytracePass->GetOutputTexture();
		const bool accumulate   = g_accumulationEnabled.Get() != 0;
		Texture& tonemapInput   = accumulate ? m_accumulationPass->GetDisplayBuffer() : raytraceOutput;

		const GraphResourceHandle raytraceOutputHandle = m_renderGraph.Import(raytraceOutput, "Raytrace Output");
		const GraphResourceHandle tonemapInputHandle   = m_renderGraph.Import(tonemapInput, "Tonemap Input");
		const GraphResourceHandle tonemapOutputHandle  = m_renderGraph.Import(m_postProcessPass->GetOutputBuffer(), "PostProcess Output");
		const GraphResourceHandle backBufferHandle     = m_renderGraph.Import(backBuffer, "Back Buffer");

		// The presented image is the graph's sink: culling walks backwards from here.
		m_renderGraph.MarkExternallyRead(backBufferHandle);

		m_renderGraph.AddPass("Raytrace Technique",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Write(raytraceOutputHandle, GraphAccess::ComputeWrite);
				// A guiding technique reads the VXPG products; this declaration is
				// what keeps their producers alive through culling.
				DeclareGuidingReads(pass);
			},
			[this]() { m_raytracePass->Render(); });

		if (accumulate)
		{
			m_renderGraph.AddPass("Accumulation",
				[&](RenderGraphPassBuilder& pass)
				{
					pass.Read(raytraceOutputHandle, GraphAccess::ComputeRead);
					pass.Write(tonemapInputHandle, GraphAccess::ComputeWrite);
				},
				[this, &raytraceOutput]() { m_accumulationPass->Render(raytraceOutput); });
		}

		m_renderGraph.AddPass("PostProcess",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(tonemapInputHandle, GraphAccess::ComputeRead);
				pass.Write(tonemapOutputHandle, GraphAccess::ComputeWrite);
			},
			[this, &tonemapInput, postProcessParams]()
			{
				m_postProcessPass->Dispatch(tonemapInput, postProcessParams);
			});

		m_renderGraph.AddPass("Present Copy",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(tonemapOutputHandle, GraphAccess::CopySource);
				pass.Write(backBufferHandle, GraphAccess::CopyDestination);
			},
			[this, &backBuffer]() { m_postProcessPass->CopyToBackBuffer(backBuffer); });

		// Readback for a screenshot armed by Tick(). A node so the copy-source state
		// it needs is declared rather than inherited from whatever ran before it.
		if (m_screenshotManager->IsCaptureDue())
		{
			m_renderGraph.AddPass("Screenshot Readback",
				[&](RenderGraphPassBuilder& pass)
				{
					pass.NeverCull();
					pass.Read(tonemapOutputHandle, GraphAccess::CopySource);
				},
				[this]()
				{
					m_screenshotManager->RecordCopy(m_postProcessPass->GetOutputBuffer().GetUnderlyingResource());
				});
		}

		// ImGui and the present transition still expect a render target.
		m_renderGraph.AddPass("Back Buffer To Render Target",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.NeverCull();
				pass.Write(backBufferHandle, GraphAccess::RenderTarget);
			},
			nullptr);

		m_renderGraph.Compile();
		m_renderGraph.Execute(CommandContext::Get());

		// Restore main descriptor heap for ImGui (post-process pass may have changed it)
		ID3D12DescriptorHeap* mainHeaps[] = { GlobalDescriptorHeap::Get().GetHeap() };
		m_d3d12CommandList->SetDescriptorHeaps(_countof(mainHeaps), mainHeaps);
	}

	if (!m_headless)
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_d3d12CommandList.Get());
	
	backBuffer.TransitionChecked(m_d3d12CommandList.Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT);

	ID3D12CommandList* const commandLists[] = { m_d3d12CommandList.Get() };

	CommandContext::Get().Close();

	m_graphicsDevice->GetCommandQueue()->ExecuteCommandLists(_countof(commandLists), commandLists);
	ResourceStateTracker::Get().OnExecuteCommandLists();
	UINT presentFlags = m_graphicsDevice->IsTearingSupported() ? DXGI_PRESENT_ALLOW_TEARING : 0; // TODO: do not check every time

	ThrowIfFailed(m_graphicsDevice->GetSwapChain()->Present(0, presentFlags));

	FlushCommandQueue();
	ResetCommandList();

	// Same guarantee the screenshot readback relies on: the GPU is done, so the
	// timestamp readback buffer holds this frame's values. The dump runs after it
	// so a single one-shot dump carries the node costs of the frame it describes.
	m_renderGraph.ResolveTimings();
	DumpRenderGraphIfRequested();

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
	assert(m_graphicsDevice->GetSwapChain() && "Attempted to resize window without swap chain.");
	
	auto& window = Window::Get();

	FlushCommandQueue();

	for (int i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		m_backBufferTextures[i].reset();
		m_d3d12RenderTargets[i].Reset();
	}

	int swapChainFlags = m_graphicsDevice->CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	ThrowIfFailed(m_graphicsDevice->GetSwapChain()->ResizeBuffers(
		Constants::Graphics::NUM_FRAMES,
		window.GetWidth(),
		window.GetHeight(),
		m_backBufferFormat,
		swapChainFlags));

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_graphicsDevice->GetSwapChain()->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		g_device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);
		m_backBufferTextures[i] = std::make_unique<Texture>(g_device, m_d3d12RenderTargets[i],
			D3D12_RESOURCE_STATE_PRESENT, L"Back Buffer " + std::to_wstring(i));
		rtvHandle.Offset(1, m_rtvDescriptorSize);
	}
	
	m_depthStencilTexture.reset();
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

void Renderer::CreateCommandList()
{
	ThrowIfFailed(g_device->CreateCommandList(
		0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_graphicsDevice->GetCommandAllocator(m_graphicsDevice->GetFrameIndex()).Get(),
		nullptr,
		IID_PPV_ARGS(&m_d3d12CommandList)));

	CommandContext::Get().Bind(m_d3d12CommandList.Get());
	CommandContext::Get().Close();
}

void Renderer::ResetCommandList() const
{
	auto& allocator = m_graphicsDevice->GetCommandAllocator(m_graphicsDevice->GetFrameIndex());
	ThrowIfFailed(allocator->Reset());
	ThrowIfFailed(m_d3d12CommandList->Reset(allocator.Get(), m_pipelineStateObject.Get()));
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
		ThrowIfFailed(m_graphicsDevice->GetSwapChain()->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		g_device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);
		m_backBufferTextures[i] = std::make_unique<Texture>(g_device, m_d3d12RenderTargets[i],
			D3D12_RESOURCE_STATE_PRESENT, L"Back Buffer " + std::to_wstring(i));

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

	m_depthStencilTexture = std::make_unique<Texture>(g_device, m_depthStencilBuffer,
		D3D12_RESOURCE_STATE_COMMON, L"Depth Stencil");

	D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc;
	viewDesc.Flags = D3D12_DSV_FLAG_NONE;
	viewDesc.Format = m_depthStencilFormat;
	viewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	viewDesc.Texture2D.MipSlice = 0;

	g_device->CreateDepthStencilView(
		m_depthStencilBuffer.Get(),
		&viewDesc,
		m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	m_depthStencilTexture->TransitionChecked(m_d3d12CommandList.Get(),
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_STATE_DEPTH_WRITE);
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

	g_device->CreateConstantBufferView(&cbvDesc,
		GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::CameraMatrices));
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
	cbvRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::CameraMatrices);

	D3D12_DESCRIPTOR_RANGE rtRange;
	rtRange.BaseShaderRegister = 0;
	rtRange.NumDescriptors = 1;
	rtRange.RegisterSpace = 0;
	rtRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	rtRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::RaytraceOutput);

	D3D12_DESCRIPTOR_RANGE tlasRange;
	tlasRange.BaseShaderRegister = 0;
	tlasRange.NumDescriptors = 1;
	tlasRange.RegisterSpace = 0;
	tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	tlasRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Tlas);

	D3D12_DESCRIPTOR_RANGE vertexRange;
	vertexRange.BaseShaderRegister = 1;
	vertexRange.NumDescriptors = 1;
	vertexRange.RegisterSpace = 0;
	vertexRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	vertexRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Vertices);
	
	D3D12_DESCRIPTOR_RANGE indexRange;
	indexRange.BaseShaderRegister = 2;
	indexRange.NumDescriptors = 1;
	indexRange.RegisterSpace = 0;
	indexRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	indexRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Indices);
	
	D3D12_DESCRIPTOR_RANGE textureRange;
	textureRange.BaseShaderRegister = 3;
	textureRange.NumDescriptors = Constants::Graphics::MAX_TEXTURES;
	textureRange.RegisterSpace = 0;
	textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	textureRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::MaterialTextures);

	// u1 = occupancy, u2 = packed irradiance, u3 = vpl count (three contiguous slots)
	D3D12_DESCRIPTOR_RANGE voxelOccupancyRange;
	voxelOccupancyRange.BaseShaderRegister = 1;
	voxelOccupancyRange.NumDescriptors = 3;
	voxelOccupancyRange.RegisterSpace = 0;
	voxelOccupancyRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	voxelOccupancyRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelOccupancy);

	// u4 = ShadingPoints G-buffer (debug overlay reads it by screen pixel)
	D3D12_DESCRIPTOR_RANGE shadingPointsRange;
	shadingPointsRange.BaseShaderRegister = 4;
	shadingPointsRange.NumDescriptors = 1;
	shadingPointsRange.RegisterSpace = 0;
	shadingPointsRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	shadingPointsRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::ShadingPoints);

	// u7 = superpixel index, u8 = superpixel representative center (debug views 15/16)
	D3D12_DESCRIPTOR_RANGE superpixelRange;
	superpixelRange.BaseShaderRegister = 7;
	superpixelRange.NumDescriptors = 2;
	superpixelRange.RegisterSpace = 0;
	superpixelRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	superpixelRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::SuperpixelIndex);

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
	m_graphicsDevice->FlushCommandQueue();
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

	const auto handle = GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::MaterialTextures,
		texture->GetTextureIndex());

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

	g_device->CreateShaderResourceView(vertex_buffer->GetUnderlyingResource().Get(), &srv_desc,
		GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::Vertices));
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

	g_device->CreateShaderResourceView(index_buffer->GetUnderlyingResource().Get(), &srv_desc,
		GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::Indices));
}

void Renderer::InitializeEditorUI()
{
	m_editorUI = std::make_shared<EditorUI>();
	m_editorUI->Initialize(g_device, m_graphicsDevice->GetCommandQueue(), GlobalDescriptorHeap::Get().GetHeap());
	m_editorUI->SetCamera(m_camera);
	m_editorUI->SetScene(m_scene);
	m_editorUI->SetAccumulationPass(m_accumulationPass);
	m_editorUI->SetStatesManager(m_statesManager);
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
	m_editorUI->SetRenderGraphTimingsGetter([this]() -> const std::vector<RenderGraph::PassTiming>& {
		return m_renderGraph.GetTimings();
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

	// The outgoing scene's textures die with m_scene below; their slots go back to
	// the free list holding null views so nothing in the heap outlives its resource.
	GlobalDescriptorHeap::Get().ReleaseMaterialTextureSlots();
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
	if (m_statesManager)
		m_statesManager->OnSceneChanged(ExtractModelName(path));
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
	newPass->Initialize(g_device, m_d3d12CommandList, m_scene, m_randomBuffer->GetUnderlyingResource(), m_passConstants);
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

std::vector<std::string> Renderer::GetStateNames() const
{
	std::vector<std::string> names;
	if (m_statesManager)
		for (const State& state : m_statesManager->GetStatesForCurrentScene())
			names.push_back(state.name);
	return names;
}

bool Renderer::GoToState(const std::string& name)
{
	return m_statesManager && m_statesManager->GoToStateByName(name);
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
	g_guidingTreeWeightMode.Set(static_cast<int32_t>(config.treeWeightMode));
	g_guidingSecondBounce.Set(config.secondBounce ? 1 : 0);
	g_guidingOneSampleMis.Set(config.oneSampleMis ? 1 : 0);
	g_guidingOneSampleAdaptiveQ.Set(config.oneSampleAdaptiveQ ? 1 : 0);
	g_injectionReuseGi.Set(config.injectionReuse ? 1 : 0);
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

bool Renderer::FrameUsesVoxelGuiding() const
{
	// One bit, not a stage ladder: it only gates the eager world-space bake, which
	// has to run before the graph imports because a grid resize recreates the very
	// textures those imports capture. Everything past that is the graph's call.
	return m_rasterize
		? RasterDebugViewUsesVoxelGuiding(g_rasterizationDebugMode.Get())
		: (m_raytracePass && m_raytracePass->UsesVoxelGuiding());
}

void Renderer::BuildVxpgGraph()
{
	m_vxpg = VxpgGraphHandles{};

	if (!FrameUsesVoxelGuiding() || !m_voxelizationPass || !m_scene)
		return;

	// Stage 1: geometry bake (rebakes only when invalidated) + per-frame
	// injection-accumulator clear.
	// Debug views suppress the BSDF subtree whose bounce writes the VPL data,
	// which would starve the guide within two frames — fall back to the
	// dedicated injection trace whenever one is active. The symmetric baseline
	// (view 15) is exempt: its BSDF sample always traces and writes VPLs, so
	// reuse stays on and its frame cost matches the full integrator's.
	const bool reuseGiVpl = g_injectionReuseGi.Get() != 0 &&
		(g_guidingDebugView.Get() == GuidingDebugView::None ||
		 g_guidingDebugView.Get() == GuidingDebugView::SymmetricBsdfBaseline);
	m_voxelizationPass->SetRuntimeParams(
		g_voxelInjectUseAvg.Get() != 0,
		g_voxelHeatScale.Get(),
		reuseGiVpl);

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
		// Faithful config: wipe the injection accumulators up front, the
		// injection trace refills them this frame. Reuse config wipes after
		// the guiding build instead (below) — the build passes consume last
		// frame's GI-written VPL data first (ADR 0009).
		if (!reuseGiVpl)
			m_voxelizationPass->DispatchFrameClear();
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

	// Imports last one frame (Reset drops them), so a texture the bake above just
	// recreated can never be reached through a stale pointer. Invalid handles are
	// dropped by the builder, so a pass that does not exist declares nothing.
	auto importRaw    = [&](ID3D12Resource* r, const char* name) { return m_renderGraph.ImportRaw(r, name); };
	auto importBuffer = [&](auto* buffer, const char* name)
	{
		return buffer ? m_renderGraph.Import(*buffer, name) : InvalidGraphResource;
	};

	m_vxpg.voxelOccupancy  = importRaw(m_voxelizationPass->GetOccupancyTexture().Get(), "VXPG VoxelOccupancy");
	m_vxpg.voxelIrradiance = importRaw(m_voxelizationPass->GetIrradianceTexture().Get(), "VXPG VoxelIrradiance");
	m_vxpg.voxelVplCount   = importRaw(m_voxelizationPass->GetVplCountTexture().Get(), "VXPG VoxelVplCount");

	if (m_vbufferPass)
		m_vxpg.vbuffer = importRaw(m_vbufferPass->GetVBufferTexture().Get(), "VXPG VBuffer");
	if (m_lightInjectionPass)
	{
		m_vxpg.shadingPoints       = importRaw(m_lightInjectionPass->GetShadingPointsTexture().Get(), "VXPG ShadingPoints");
		m_vxpg.voxelRepresentative = importRaw(m_lightInjectionPass->GetVoxelRepresentativeTexture().Get(), "VXPG VoxelRepresentative");
		m_vxpg.vplPosition         = importRaw(m_lightInjectionPass->GetVplPositionTexture().Get(), "VXPG VplPosition");
	}
	if (m_voxelGuidingBuildPass)
	{
		m_vxpg.counters           = importBuffer(m_voxelGuidingBuildPass->GetCountersBuffer(), "VXPG Counters");
		m_vxpg.compactIds         = importBuffer(m_voxelGuidingBuildPass->GetCompactIdsBuffer(), "VXPG CompactIds");
		m_vxpg.inverseIndex       = importBuffer(m_voxelGuidingBuildPass->GetInverseIndexBuffer(), "VXPG InverseIndex");
		m_vxpg.compactLightPoints = importBuffer(m_voxelGuidingBuildPass->GetCompactVoxelLightPointsBuffer(), "VXPG CompactLightPoints");
		m_vxpg.premulIrradiance   = importBuffer(m_voxelGuidingBuildPass->GetPremulIrradianceBuffer(), "VXPG PremulIrradiance");
		m_vxpg.liveBoundMin       = importBuffer(m_voxelGuidingBuildPass->GetLiveBoundMinBuffer(), "VXPG LiveBoundMin");
		m_vxpg.liveBoundMax       = importBuffer(m_voxelGuidingBuildPass->GetLiveBoundMaxBuffer(), "VXPG LiveBoundMax");
	}
	if (m_fingerprintPass)
		m_vxpg.voxelFingerprints = importBuffer(m_fingerprintPass->GetVoxelFingerprintsBuffer(), "VXPG VoxelFingerprints");
	if (m_clusterPass)
	{
		m_vxpg.clusterAssignments    = importBuffer(m_clusterPass->GetVoxelClusterAssignmentsBuffer(), "VXPG ClusterAssignments");
		m_vxpg.clusterSeedCompactIds = importBuffer(m_clusterPass->GetClusterSeedCompactIdsBuffer(), "VXPG ClusterSeedCompactIds");
	}
	if (m_superpixelBuildPass)
	{
		m_vxpg.superpixelIndex       = importRaw(m_superpixelBuildPass->GetIndexResource(), "VXPG SuperpixelIndex");
		m_vxpg.superpixelCenter      = importRaw(m_superpixelBuildPass->GetCenterResource(), "VXPG SuperpixelCenter");
		m_vxpg.superpixelCounter     = importRaw(m_superpixelBuildPass->GetCounterResource(), "VXPG SuperpixelCounter");
		m_vxpg.superpixelGathered    = importRaw(m_superpixelBuildPass->GetGatheredResource(), "VXPG SuperpixelGathered");
		m_vxpg.superpixelFuzzyWeight = importRaw(m_superpixelBuildPass->GetFuzzyWeightResource(), "VXPG SuperpixelFuzzyWeight");
		m_vxpg.superpixelFuzzyIndex  = importRaw(m_superpixelBuildPass->GetFuzzyIndexResource(), "VXPG SuperpixelFuzzyIndex");
	}
	if (m_clusterVisibilityPass)
	{
		m_vxpg.clusterVisibilityMask = importRaw(m_clusterVisibilityPass->GetMaskResource(), "VXPG ClusterVisibilityMask");
		m_vxpg.avgVisibility         = importBuffer(m_clusterVisibilityPass->GetAvgVisibilityBuffer(), "VXPG AvgVisibility");
	}
	if (m_lightTreePass)
	{
		m_vxpg.lightTreeNodes         = importBuffer(m_lightTreePass->GetNodesBuffer(), "VXPG LightTreeNodes");
		m_vxpg.lightTreeCompactToLeaf = importBuffer(m_lightTreePass->GetCompactToLeafBuffer(), "VXPG LightTreeCompactToLeaf");
		m_vxpg.lightTreeClusterRoots  = importBuffer(m_lightTreePass->GetClusterRootsBuffer(), "VXPG LightTreeClusterRoots");
		m_vxpg.superpixelClusterHeap  = importBuffer(m_lightTreePass->GetSuperpixelClusterHeapBuffer(), "VXPG SuperpixelClusterHeap");
	}

	// Every node below is added unconditionally; the ones whose products nothing
	// reads this frame are culled. Buffers a pass only hands between its own
	// kernels stay hand-barriered — those hazards are intra-pass, which a single
	// node cannot express.
	constexpr GraphAccess kUavRead  = GraphAccess::UnorderedAccessRead;
	constexpr GraphAccess kUavWrite = GraphAccess::ComputeWrite;

	// Stage 2: shared VBuffer (one jittered primary per pixel, ADR 0004), then
	// light injection reconstructing its first vertex from it (also emits the
	// ShadingPoints G-buffer).
	if (m_vbufferPass)
	{
		m_renderGraph.AddPass("VXPG VBuffer",
			[&](RenderGraphPassBuilder& pass) { pass.Write(m_vxpg.vbuffer, kUavWrite); },
			[this]() { m_vbufferPass->Render(); });
	}
	if (m_lightInjectionPass)
	{
		m_renderGraph.AddPass("VXPG LightInjection",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.vbuffer, kUavRead);
				pass.Read(m_vxpg.voxelOccupancy, kUavRead);
				pass.Write(m_vxpg.shadingPoints, kUavWrite);
				pass.Write(m_vxpg.voxelRepresentative, kUavWrite);
				pass.Write(m_vxpg.vplPosition, kUavWrite);
				pass.Write(m_vxpg.voxelIrradiance, kUavWrite);
				pass.Write(m_vxpg.voxelVplCount, kUavWrite);
			},
			[this]() { m_lightInjectionPass->Render(); });
	}

	// Stage 3: build the guiding distribution from the injected voxels
	// (reload baked bounds -> compact -> CDF).
	if (m_voxelGuidingBuildPass)
	{
		m_renderGraph.AddPass("VXPG GuidingBuild",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.voxelIrradiance, kUavRead);
				pass.Read(m_vxpg.voxelVplCount, kUavRead);
				pass.Read(m_vxpg.voxelRepresentative, kUavRead);
				pass.Write(m_vxpg.counters, kUavWrite);
				pass.Write(m_vxpg.compactIds, kUavWrite);
				pass.Write(m_vxpg.inverseIndex, kUavWrite);
				pass.Write(m_vxpg.compactLightPoints, kUavWrite);
				pass.Write(m_vxpg.premulIrradiance, kUavWrite);
				pass.Write(m_vxpg.liveBoundMin, kUavWrite);
				pass.Write(m_vxpg.liveBoundMax, kUavWrite);
			},
			[this]()
			{
				m_voxelGuidingBuildPass->Run(
					m_lightInjectionPass ? m_lightInjectionPass->GetVoxelRepresentativeTexture().Get() : nullptr);
			});
	}

	// Stage 4: fingerprint every lit voxel (128 stratified screen representatives
	// -> per-voxel visibility mask via inline shadow rays).
	if (m_fingerprintPass && m_scene)
	{
		m_renderGraph.AddPass("VXPG Fingerprint",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.shadingPoints, kUavRead);
				pass.Read(m_vxpg.counters, kUavRead);
				pass.Read(m_vxpg.compactLightPoints, kUavRead);
				pass.Write(m_vxpg.voxelFingerprints, kUavWrite);
			},
			[this]()
			{
				auto tlas = m_scene->GetAccelerationStructures()->GetTopLevelAS().p_result;
				m_fingerprintPass->Run(tlas ? tlas->GetGPUVirtualAddress() : 0,
					m_passConstants->data.frameIndex);
			});
	}

	// Stage 5: k-means++ cluster the fingerprinted voxels into 32 supervoxels.
	if (m_clusterPass)
	{
		m_renderGraph.AddPass("VXPG Cluster",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.voxelFingerprints, kUavRead);
				pass.Read(m_vxpg.compactIds, kUavRead);
				pass.Read(m_vxpg.premulIrradiance, kUavRead);
				pass.Write(m_vxpg.clusterAssignments, kUavWrite);
				pass.Write(m_vxpg.clusterSeedCompactIds, kUavWrite);
			},
			[this]() { m_clusterPass->Run(m_passConstants->data.frameIndex); });
	}

	// Stage 6: SLIC superpixel clustering over the ShadingPoints G-buffer.
	if (m_superpixelBuildPass)
	{
		m_renderGraph.AddPass("VXPG Superpixel",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.shadingPoints, kUavRead);
				pass.Write(m_vxpg.superpixelIndex, kUavWrite);
				pass.Write(m_vxpg.superpixelCenter, kUavWrite);
				pass.Write(m_vxpg.superpixelCounter, kUavWrite);
				pass.Write(m_vxpg.superpixelGathered, kUavWrite);
				pass.Write(m_vxpg.superpixelFuzzyWeight, kUavWrite);
				pass.Write(m_vxpg.superpixelFuzzyIndex, kUavWrite);
			},
			[this]()
			{
				m_superpixelBuildPass->Run(
					m_lightInjectionPass ? m_lightInjectionPass->GetShadingPointsTexture().Get() : nullptr,
					g_superpixelWeight.Get(), g_superpixelPosNormalizer.Get());
			});
	}

	// Stage 7: per-superpixel x per-cluster soft visibility (cvis).
	if (m_clusterVisibilityPass)
	{
		m_renderGraph.AddPass("VXPG ClusterVisibility",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.vplPosition, kUavRead);
				pass.Read(m_vxpg.vbuffer, kUavRead);
				pass.Read(m_vxpg.clusterAssignments, kUavRead);
				pass.Read(m_vxpg.superpixelIndex, kUavRead);
				pass.Read(m_vxpg.superpixelGathered, kUavRead);
				pass.Read(m_vxpg.superpixelCounter, kUavRead);
				pass.Write(m_vxpg.clusterVisibilityMask, kUavWrite);
				pass.Write(m_vxpg.avgVisibility, kUavWrite);
			},
			[this]() { m_clusterVisibilityPass->Run(m_passConstants->data.frameIndex); });
	}

	// Stage 8: bottom light tree (Karras LBVH over lit voxels: encode -> sort ->
	// initial -> internal -> merge).
	if (m_lightTreePass)
	{
		m_renderGraph.AddPass("VXPG LightTree",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.Read(m_vxpg.clusterAssignments, kUavRead);
				pass.Read(m_vxpg.counters, kUavRead);
				pass.Read(m_vxpg.compactIds, kUavRead);
				pass.Read(m_vxpg.compactLightPoints, kUavRead);
				pass.Read(m_vxpg.premulIrradiance, kUavRead);
				pass.Read(m_vxpg.avgVisibility, kUavRead);
				pass.Write(m_vxpg.lightTreeNodes, kUavWrite);
				pass.Write(m_vxpg.lightTreeCompactToLeaf, kUavWrite);
				pass.Write(m_vxpg.lightTreeClusterRoots, kUavWrite);
				pass.Write(m_vxpg.superpixelClusterHeap, kUavWrite);
			},
			[this]() { m_lightTreePass->Run(); });
	}

	// Reuse config (ADR 0009): the build above consumed last frame's VPL data;
	// wipe the accumulators now so the guided GI raygen refills them fresh. Its
	// consumer is next frame's build, which culling cannot see.
	if (reuseGiVpl)
	{
		m_renderGraph.AddPass("VXPG InjectionClear",
			[&](RenderGraphPassBuilder& pass)
			{
				pass.NeverCull();
				pass.Write(m_vxpg.voxelIrradiance, kUavWrite);
				pass.Write(m_vxpg.voxelVplCount, kUavWrite);
			},
			[this]() { m_voxelizationPass->DispatchFrameClear(/*emitTailBarriers*/ false); });
	}
}

void Renderer::DeclareGuidingReads(RenderGraphPassBuilder& pass)
{
	// Only a guiding technique touches these; a plain path tracer declares nothing
	// and every VXPG node is culled.
	if (!m_raytracePass || !m_raytracePass->UsesVoxelGuiding())
		return;

	constexpr GraphAccess kUavRead = GraphAccess::UnorderedAccessRead;

	// Read through the global descriptor table.
	pass.Read(m_vxpg.voxelOccupancy, kUavRead);
	pass.Read(m_vxpg.voxelIrradiance, kUavRead);
	pass.Read(m_vxpg.voxelVplCount, kUavRead);
	pass.Read(m_vxpg.voxelRepresentative, kUavRead);
	pass.Read(m_vxpg.shadingPoints, kUavRead);
	pass.Read(m_vxpg.vbuffer, kUavRead);
	pass.Read(m_vxpg.superpixelIndex, kUavRead);
	pass.Read(m_vxpg.superpixelCenter, kUavRead);
	pass.Read(m_vxpg.superpixelFuzzyWeight, kUavRead);
	pass.Read(m_vxpg.superpixelFuzzyIndex, kUavRead);
	pass.Read(m_vxpg.clusterVisibilityMask, kUavRead);

	// Read as root UAVs (GuidedPathTracingPass root parameters 8-19).
	pass.Read(m_vxpg.counters, kUavRead);
	pass.Read(m_vxpg.compactIds, kUavRead);
	pass.Read(m_vxpg.inverseIndex, kUavRead);
	pass.Read(m_vxpg.voxelFingerprints, kUavRead);
	pass.Read(m_vxpg.clusterAssignments, kUavRead);
	pass.Read(m_vxpg.clusterSeedCompactIds, kUavRead);
	pass.Read(m_vxpg.lightTreeNodes, kUavRead);
	pass.Read(m_vxpg.lightTreeCompactToLeaf, kUavRead);
	pass.Read(m_vxpg.lightTreeClusterRoots, kUavRead);
	pass.Read(m_vxpg.superpixelClusterHeap, kUavRead);
	pass.Read(m_vxpg.liveBoundMin, kUavRead);
	pass.Read(m_vxpg.liveBoundMax, kUavRead);
}

void Renderer::DeclareRasterDebugViewReads(RenderGraphPassBuilder& pass)
{
	// The raster draws are not a node yet (phase 5), so this stands in for them:
	// whatever the active debug view samples is what keeps VXPG stages alive.
	constexpr GraphAccess kUavRead = GraphAccess::UnorderedAccessRead;

	switch (g_rasterizationDebugMode.Get())
	{
	case RasterDebugMode::VoxelOccupancy:
	case RasterDebugMode::Supervoxels:
		pass.Read(m_vxpg.voxelOccupancy, kUavRead);
		break;
	case RasterDebugMode::VoxelIrradiance:
		pass.Read(m_vxpg.voxelIrradiance, kUavRead);
		pass.Read(m_vxpg.voxelVplCount, kUavRead);
		break;
	case RasterDebugMode::ShadingPointsNormal:
	case RasterDebugMode::ShadingPointsPos:
		pass.Read(m_vxpg.shadingPoints, kUavRead);
		break;
	case RasterDebugMode::SuperpixelId:
		pass.Read(m_vxpg.superpixelIndex, kUavRead);
		break;
	case RasterDebugMode::SuperpixelRepresentative:
		pass.Read(m_vxpg.superpixelIndex, kUavRead);
		pass.Read(m_vxpg.superpixelCenter, kUavRead);
		break;
	default:
		break;
	}
}

void Renderer::DumpRenderGraphIfRequested()
{
	if (g_dumpRenderGraph.Get() == 0)
		return;

	spdlog::info("[RDG] frame passes:\n{}", m_renderGraph.DumpPasses());
	spdlog::info("[RDG] synthesized barriers:\n{}", m_renderGraph.DumpBarriers());

	// Last resolved frame's node costs, when rdg.timings is on — the same numbers
	// the ImGui table shows, so a headless run can attribute a regression too.
	std::string timings;
	for (const auto& timing : m_renderGraph.GetTimings())
		timings += fmt::format("    {:<32} {:.3f} ms\n", timing.name, timing.milliseconds);
	if (!timings.empty())
		spdlog::info("[RDG] node GPU cost:\n{}", timings);

	g_dumpRenderGraph.Set(0); // one-shot: a per-frame dump is unreadable
}

void Renderer::WriteVoxelUavsToGlobalHeap()
{
	GlobalDescriptorHeap& heap = GlobalDescriptorHeap::Get();

	if (!m_voxelizationPass)
	{
		heap.ClearSlot(GlobalDescriptor::VoxelOccupancy);
		heap.ClearSlot(GlobalDescriptor::VoxelIrradiance);
		heap.ClearSlot(GlobalDescriptor::VoxelVplCount);
		return;
	}

	m_voxelizationPass->WriteOccupancyUavTo(heap.CpuHandle(GlobalDescriptor::VoxelOccupancy));
	m_voxelizationPass->WriteIrradianceUavTo(heap.CpuHandle(GlobalDescriptor::VoxelIrradiance));
	m_voxelizationPass->WriteVplCountUavTo(heap.CpuHandle(GlobalDescriptor::VoxelVplCount));
}

void Renderer::WriteSuperpixelUavsToGlobalHeap()
{
	GlobalDescriptorHeap& heap = GlobalDescriptorHeap::Get();

	if (!m_superpixelBuildPass)
	{
		heap.ClearSlot(GlobalDescriptor::SuperpixelIndex);
		heap.ClearSlot(GlobalDescriptor::SuperpixelCenter);
		heap.ClearSlot(GlobalDescriptor::FuzzyWeight);
		heap.ClearSlot(GlobalDescriptor::FuzzyIndex);
		return;
	}

	m_superpixelBuildPass->WriteIndexUavTo(heap.CpuHandle(GlobalDescriptor::SuperpixelIndex));
	m_superpixelBuildPass->WriteCenterUavTo(heap.CpuHandle(GlobalDescriptor::SuperpixelCenter));
	m_superpixelBuildPass->WriteFuzzyWeightUavTo(heap.CpuHandle(GlobalDescriptor::FuzzyWeight));
	m_superpixelBuildPass->WriteFuzzyIndexUavTo(heap.CpuHandle(GlobalDescriptor::FuzzyIndex));
}

void Renderer::WriteClusterVisibilityUavsToGlobalHeap()
{
	GlobalDescriptorHeap& heap = GlobalDescriptorHeap::Get();

	auto writeUav = [&](GlobalDescriptor slot, ID3D12Resource* res, DXGI_FORMAT fmt)
	{
		if (!res)
		{
			heap.ClearSlot(slot);
			return;
		}

		D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
		uav.Format        = fmt;
		uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		g_device->CreateUnorderedAccessView(res, nullptr, &uav, heap.CpuHandle(slot));
	};

	writeUav(GlobalDescriptor::SpixelGathered,
		m_superpixelBuildPass ? m_superpixelBuildPass->GetGatheredResource() : nullptr, DXGI_FORMAT_R32G32_SINT);
	writeUav(GlobalDescriptor::SpixelCounter,
		m_superpixelBuildPass ? m_superpixelBuildPass->GetCounterResource() : nullptr, DXGI_FORMAT_R32_UINT);
	writeUav(GlobalDescriptor::ClusterVisibilityMask,
		m_clusterVisibilityPass ? m_clusterVisibilityPass->GetMaskResource() : nullptr, DXGI_FORMAT_R32_UINT);
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

	m_skyboxTexture = std::make_unique<Texture>(g_device, textureResource,
		D3D12_RESOURCE_STATE_COMMON, L"Skybox");

	m_skyboxTexture->TransitionChecked(m_d3d12CommandList.Get(),
		D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

	UpdateSubresources(CommandContext::Get().GetCommandList(), textureResource.Get(), uploadBuffer.Get(),
		0, 0, static_cast<UINT>(subresources.size()), subresources.data());

	m_skyboxTexture->TransitionChecked(m_d3d12CommandList.Get(),
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	auto desc = textureResource->GetDesc();
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = desc.Format;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = desc.MipLevels;

	g_device->CreateShaderResourceView(textureResource.Get(), &srvDesc,
		GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::Skybox));

	ExecuteCommandsAndReset();
	spdlog::info("Skybox loaded successfully.");
}

void Renderer::OnShaderReload()
{
	spdlog::info("Reloading shaders...");
	ResourceManager::Get().RecompileAllShaders();
	
	CommandContext::Get().Close();
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_graphicsDevice->GetCommandQueue()->ExecuteCommandLists(_countof(commandLists), commandLists);
	ResourceStateTracker::Get().OnExecuteCommandLists();

	FlushCommandQueue();

	spdlog::info("Creating pipeline state for new shaders...");
	CreatePipelineState();
	// Every cached compute PSO at once — the VXPG passes hold pointer-stable
	// programs, so none of them needs its own reload path.
	ShaderProgramCache::Get().RebuildAll();
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

	std::shared_ptr<Texture> texture = std::make_shared<Texture>(g_device, texture_resource,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	CreateTextureSRV(texture);

	CommandContext::Get().Close();
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_graphicsDevice->GetCommandQueue()->ExecuteCommandLists(_countof(commandLists), commandLists);
	ResourceStateTracker::Get().OnExecuteCommandLists();
	FlushCommandQueue();
	ResetCommandList();

	m_textures.push_back(texture);
	
	return texture;
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
	CommandContext::Get().Close();
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_graphicsDevice->GetCommandQueue()->ExecuteCommandLists(_countof(commandLists), commandLists);
	ResourceStateTracker::Get().OnExecuteCommandLists();
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
