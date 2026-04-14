#include "pch.h"
#include "EditorUI.h"

#include <commdlg.h>

#include "imgui.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"
#include "ImGuizmo.h"

#include "Camera.h"
#include "Constants.h"
#include "FrameAccumulationPass.h"
#include "RaytracePass.h"
#include "SceneResources/Scene.h"
#include "SceneResources/LightData.h"
#include "Window.h"

void EditorUI::Initialize(
	Microsoft::WRL::ComPtr<ID3D12Device5> device,
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue,
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap)
{
	ImGui_ImplWin32_EnableDpiAwareness();
	float mainScaleImGui = ImGui_ImplWin32_GetDpiScaleForMonitor(::MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY));

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui::StyleColorsDark();

	ImGuiStyle& style = ImGui::GetStyle();
	style.ScaleAllSizes(mainScaleImGui);

	ImGui_ImplWin32_Init(Window::Get().GetHandle());

	ImGui_ImplDX12_InitInfo init_info = {};
	init_info.Device = device.Get();
	init_info.CommandQueue = commandQueue.Get();
	init_info.NumFramesInFlight = Constants::Graphics::NUM_FRAMES;
	init_info.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	init_info.DSVFormat = DXGI_FORMAT_D32_FLOAT;

	init_info.LegacySingleSrvCpuDescriptor.ptr = srvHeap->GetCPUDescriptorHandleForHeapStart().ptr;
	init_info.LegacySingleSrvGpuDescriptor.ptr = srvHeap->GetGPUDescriptorHandleForHeapStart().ptr;
	init_info.SrvDescriptorHeap = srvHeap.Get();

	ImGui_ImplDX12_Init(&init_info);
	spdlog::info("ImGui initialized successfully.");
}

void EditorUI::Shutdown()
{
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

void EditorUI::BeginFrame()
{
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();
	ImGuizmo::SetRect(0, 0, ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y);
	ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

	CVarSystem::Get()->DrawImguiEditor();
	DrawDebugPanel();
	DrawLightsPanel();
}

void EditorUI::EndFrame()
{
	ImGui::Render();
}

void EditorUI::DrawDebugPanel()
{
	ImGui::Begin("Debug");

	// Accumulation settings
	ImGui::SeparatorText("Temporal Accumulation");
	if (m_accumulationPass)
	{
		ImGui::Text("Frames accumulated: %u", m_accumulationPass->GetFrameCount());
		ImGui::Text("Accumulated time:   %.2fs", m_accumulationPass->GetAccumulatedTime());
	}

	auto* enabledPtr = CVarSystem::Get()->GetIntCVar(StringId("renderer.accumulation.enabled"));
	if (enabledPtr)
	{
		bool enabled = *enabledPtr != 0;
		if (ImGui::Checkbox("Enabled", &enabled))
			CVarSystem::Get()->SetCVarInt(StringId("renderer.accumulation.enabled"), enabled ? 1 : 0);
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset") && m_accumulationPass)
		m_accumulationPass->Reset();

	ImGui::DragFloat("Capture time (s)", &m_screenshotSeconds, 0.1f, 0.05f, 60.0f, "%.2f");
	if (ImGui::Button("Take Screenshot") && m_onScreenshotRequest)
		m_onScreenshotRequest(m_screenshotSeconds);
	if (m_isScreenshotPending && m_isScreenshotPending() && m_accumulationPass)
	{
		ImGui::SameLine();
		ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
			"Pending (%.2f / %.2f s)",
			m_accumulationPass->GetAccumulatedTime(),
			m_screenshotSeconds);
	}

	// Post-process settings
	ImGui::SeparatorText("Post-Process");

	auto* exposure = CVarSystem::Get()->GetFloatCVar(StringId("renderer.postprocess.exposure"));
	if (exposure && ImGui::DragFloat("Exposure", exposure, 0.05f, 0.0f, 10.0f, "%.2f"))
		CVarSystem::Get()->SetCVarFloat(StringId("renderer.postprocess.exposure"), *exposure);

	auto* contrast = CVarSystem::Get()->GetFloatCVar(StringId("renderer.postprocess.contrast"));
	if (contrast && ImGui::DragFloat("Contrast", contrast, 0.01f, 0.1f, 3.0f, "%.2f"))
		CVarSystem::Get()->SetCVarFloat(StringId("renderer.postprocess.contrast"), *contrast);

	auto* saturation = CVarSystem::Get()->GetFloatCVar(StringId("renderer.postprocess.saturation"));
	if (saturation && ImGui::DragFloat("Saturation", saturation, 0.01f, 0.0f, 2.0f, "%.2f"))
		CVarSystem::Get()->SetCVarFloat(StringId("renderer.postprocess.saturation"), *saturation);

	auto* lift = CVarSystem::Get()->GetFloatCVar(StringId("renderer.postprocess.lift"));
	if (lift && ImGui::DragFloat("Lift", lift, 0.005f, 0.0f, 0.5f, "%.3f"))
		CVarSystem::Get()->SetCVarFloat(StringId("renderer.postprocess.lift"), *lift);

	// Raytracing technique
	DrawTechniqueSection();

	// Scene
	DrawSceneSection();

	// Skybox
	DrawSkyboxSection();

	ImGui::End();
}

void EditorUI::DrawLightsPanel()
{
	if (!m_scene) return;

	auto& lights = m_scene->GetLightDataCPU();

	// Clamp selection if lights changed
	if (m_selectedLightIndex >= static_cast<int>(lights.size()))
		m_selectedLightIndex = -1;

	ImGui::Begin("Lights");

	for (int i = 0; i < static_cast<int>(lights.size()); i++)
	{
		LightData& light = lights[i];
		ImGui::PushID(i);

		char label[32];
		snprintf(label, sizeof(label), "Light %d", i);
		if (ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen))
		{
			bool dirty = false;

			if (ImGui::Button("Select Gizmo"))
				m_selectedLightIndex = (m_selectedLightIndex == i) ? -1 : i;
			if (m_selectedLightIndex == i)
			{
				ImGui::SameLine();
				ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "[Active]");
			}
			ImGui::SameLine();
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
			if (ImGui::Button("Delete"))
			{
				lights.erase(lights.begin() + i);
				m_scene->MarkLightDataDirty();
				if (m_selectedLightIndex == i)
					m_selectedLightIndex = -1;
				else if (m_selectedLightIndex > i)
					m_selectedLightIndex--;
				ImGui::PopStyleColor();
				ImGui::PopID();
				break;
			}
			ImGui::PopStyleColor();

			const char* typeNames[] = { "Directional", "Point", "Spot" };
			int lightType = static_cast<int>(light.type);
			if (ImGui::Combo("Type", &lightType, typeNames, 3))
			{
				light.type = static_cast<LightType>(lightType);
				dirty = true;
			}

			dirty |= ImGui::DragFloat3("Position", &light.position.x, 0.1f);
			dirty |= ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f);
			dirty |= ImGui::ColorEdit3("Color", &light.color.x);
			dirty |= ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 100.0f);
			dirty |= ImGui::DragFloat("Range", &light.range, 0.5f, 0.0f, 1000.0f);

			if (dirty)
				m_scene->MarkLightDataDirty();
		}

		ImGui::PopID();
	}

	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Spacing();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.6f, 0.1f, 1.0f));
	if (ImGui::Button("+", ImVec2(ImGui::GetContentRegionAvail().x, 0)))
	{
		LightData newLight{};
		newLight.type = LightType::Point;
		newLight.position = { 0.0f, 2.0f, 0.0f };
		newLight.direction = { 0.0f, -1.0f, 0.0f };
		newLight.color = { 1.0f, 1.0f, 1.0f };
		newLight.intensity = 3.0f;
		newLight.range = 20.0f;
		lights.push_back(newLight);
		m_scene->MarkLightDataDirty();
	}
	ImGui::PopStyleColor();

	// Render gizmo for selected light
	if (m_selectedLightIndex >= 0 && m_selectedLightIndex < static_cast<int>(lights.size()) && m_camera)
	{
		LightData& light = lights[m_selectedLightIndex];

		const float* view = &m_camera->GetViewMatrix()._11;
		const float* proj = &m_camera->GetProjectionMatrix()._11;

		float matrix[16];
		memset(matrix, 0, sizeof(matrix));

		ImGuizmo::OPERATION op;
		if (light.type == LightType::Directional)
		{
			using namespace DirectX;
			XMVECTOR dir = XMLoadFloat3(&light.direction);
			XMVECTOR defaultDir = XMVectorSet(0, 0, -1, 0);
			XMVECTOR axis = XMVector3Cross(defaultDir, dir);
			float dot = XMVectorGetX(XMVector3Dot(defaultDir, dir));

			XMMATRIX rot;
			if (XMVectorGetX(XMVector3LengthSq(axis)) < 1e-6f)
				rot = (dot > 0) ? XMMatrixIdentity() : XMMatrixRotationY(XM_PI);
			else
				rot = XMMatrixRotationAxis(XMVector3Normalize(axis), acosf(std::clamp(dot, -1.0f, 1.0f)));

			XMMATRIX trans = XMMatrixTranslation(light.position.x, light.position.y, light.position.z);
			XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(matrix), rot * trans);
			op = ImGuizmo::ROTATE;
		}
		else // Point (and future Spot)
		{
			matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
			matrix[12] = light.position.x;
			matrix[13] = light.position.y;
			matrix[14] = light.position.z;
			op = ImGuizmo::TRANSLATE;
		}

		ImGuizmo::Manipulate(view, proj, op, ImGuizmo::WORLD, matrix);
		if (ImGuizmo::IsUsing())
		{
			if (light.type == LightType::Directional)
			{
				using namespace DirectX;
				XMFLOAT4X4 result;
				memcpy(&result, matrix, sizeof(result));
				XMMATRIX m = XMLoadFloat4x4(&result);
				XMStoreFloat3(&light.direction, XMVector3Normalize(-m.r[2]));
			}
			else
			{
				light.position = { matrix[12], matrix[13], matrix[14] };
			}
			m_scene->MarkLightDataDirty();
		}
	}

	ImGui::End();
}

void EditorUI::DrawSkyboxSection()
{
	ImGui::SeparatorText("Skybox");

	// Display current skybox filename
	char nameUtf8[256];
	WideCharToMultiByte(CP_UTF8, 0, m_currentSkyboxName.c_str(), -1, nameUtf8, sizeof(nameUtf8), nullptr, nullptr);
	ImGui::Text("Current: %s", nameUtf8);

	if (ImGui::Button("Load Skybox") && m_onSkyboxLoad)
	{
		wchar_t filePath[MAX_PATH] = {};
		OPENFILENAMEW ofn = {};
		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = Window::Get().GetHandle();
		ofn.lpstrFilter = L"DDS Cubemap (*.dds)\0*.dds\0All Files\0*.*\0";
		ofn.lpstrFile = filePath;
		ofn.nMaxFile = MAX_PATH;
		ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

		if (GetOpenFileNameW(&ofn))
		{
			m_onSkyboxLoad(filePath);

			// Extract filename for display
			std::wstring full(filePath);
			auto pos = full.find_last_of(L"\\/");
			m_currentSkyboxName = (pos != std::wstring::npos) ? full.substr(pos + 1) : full;
		}
	}
}

void EditorUI::DrawSceneSection()
{
	ImGui::SeparatorText("Scene");

	char nameUtf8[256];
	WideCharToMultiByte(CP_UTF8, 0, m_currentSceneName.c_str(), -1, nameUtf8, sizeof(nameUtf8), nullptr, nullptr);
	ImGui::Text("Current: %s", nameUtf8);

	if (ImGui::Button("Load Scene") && m_onDifferentScenePicked)
	{
		wchar_t filePath[MAX_PATH] = {};
		OPENFILENAMEW ofn = {};
		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = Window::Get().GetHandle();
		ofn.lpstrFilter = L"glTF (*.gltf;*.glb)\0*.gltf;*.glb\0All Files\0*.*\0";
		ofn.lpstrFile = filePath;
		ofn.nMaxFile = MAX_PATH;
		ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

		if (GetOpenFileNameW(&ofn))
		{
			m_onDifferentScenePicked(filePath);

			std::wstring full(filePath);
			auto pos = full.find_last_of(L"\\/");
			m_currentSceneName = (pos != std::wstring::npos) ? full.substr(pos + 1) : full;
		}
	}
}

void EditorUI::DrawTechniqueSection()
{
	const auto& registry = RaytracePass::GetRegistry();
	if (registry.empty())
		return;

	ImGui::SeparatorText("Raytracing Technique");

	// Build label array for ImGui combo
	std::vector<const char*> names;
	names.reserve(registry.size());
	for (const auto& entry : registry)
		names.push_back(entry.name.c_str());

	if (ImGui::Combo("Technique", &m_currentTechniqueIndex, names.data(), static_cast<int>(names.size())))
	{
		if (m_onDifferentTechniquePicked)
			m_onDifferentTechniquePicked(m_currentTechniqueIndex);
	}
}
