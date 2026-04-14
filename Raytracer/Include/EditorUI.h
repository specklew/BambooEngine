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

private:
	void DrawDebugPanel();
	void DrawLightsPanel();
	void DrawSkyboxSection();

	std::shared_ptr<Scene> m_scene;
	std::shared_ptr<Camera> m_camera;
	std::shared_ptr<FrameAccumulationPass> m_accumulationPass;
	int m_selectedLightIndex = -1;
	std::function<void(const std::wstring&)> m_onSkyboxLoad;
	std::wstring m_currentSkyboxName = L"citrus_orchard_road_puresky_4k.dds";
};
