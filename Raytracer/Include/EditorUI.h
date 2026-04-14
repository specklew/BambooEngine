#pragma once

#include <functional>
#include <memory>
#include <string>

class Camera;
class Scene;
class FrameAccumulationPass;

class EditorUI
{
public:
	void Initialize(
		Microsoft::WRL::ComPtr<ID3D12Device5> device,
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue,
		Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap);

	void Shutdown();

	void BeginFrame();
	void EndFrame();

	void SetScene(std::shared_ptr<Scene> scene) { m_scene = scene; }
	void SetCamera(std::shared_ptr<Camera> camera) { m_camera = camera; }
	void SetAccumulationPass(std::shared_ptr<FrameAccumulationPass> pass) { m_accumulationPass = pass; }
	void SetSkyboxLoadCallback(std::function<void(const std::wstring&)> callback) { m_onSkyboxLoad = std::move(callback); }
	void SetOnDifferentScenePicked(std::function<void(const std::wstring&)> callback) { m_onDifferentScenePicked = std::move(callback); }
	void SetCurrentSceneName(const std::wstring& name) { m_currentSceneName = name; }
	void SetOnDifferentTechniquePicked(std::function<void(int)> callback) { m_onDifferentTechniquePicked = std::move(callback); }
	void SetScreenshotRequestCallback(std::function<void(float)> callback) { m_onScreenshotRequest = std::move(callback); }
	void SetScreenshotPendingGetter(std::function<bool()> getter)          { m_isScreenshotPending = std::move(getter); }

private:
	void DrawDebugPanel();
	void DrawLightsPanel();
	void DrawSkyboxSection();
	void DrawSceneSection();
	void DrawTechniqueSection();

	std::shared_ptr<Scene> m_scene;
	std::shared_ptr<Camera> m_camera;
	std::shared_ptr<FrameAccumulationPass> m_accumulationPass;
	int m_selectedLightIndex = -1;
	std::function<void(const std::wstring&)> m_onSkyboxLoad;
	std::wstring m_currentSkyboxName = L"qwantani_dusk_2_puresky_2k.dds";
	std::function<void(const std::wstring&)> m_onDifferentScenePicked;
	std::wstring m_currentSceneName = L"abeautifulgame.glb";
	std::function<void(int)> m_onDifferentTechniquePicked;
	int m_currentTechniqueIndex = 0;
	std::function<void(float)> m_onScreenshotRequest;
	std::function<bool()>      m_isScreenshotPending;
	float m_screenshotSeconds = 1.0f;
};
