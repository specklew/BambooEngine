#pragma once

#include <memory>

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

private:
	void DrawDebugPanel();
	void DrawLightsPanel();

	std::shared_ptr<Scene> m_scene;
	std::shared_ptr<Camera> m_camera;
	std::shared_ptr<FrameAccumulationPass> m_accumulationPass;
	int m_selectedLightIndex = -1;
};
