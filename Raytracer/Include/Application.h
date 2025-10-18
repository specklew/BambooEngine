#pragma once

#include "HighResolutionClock.h"
#include "Renderer.h"

class Application
{
public:
	Application(HINSTANCE hInstance);

	void Run();
	bool IsReady() const { return m_ready; }

private:
	friend class Window;
	
	void GameLoop();
	void OnResize();
	void OnMouseMove(WPARAM btnState, int x, int y);
	void OnKeyDown(WPARAM btnState);
	void ReportLiveObjects();
	void SetupLoggingLevel();

	bool m_ready = false;
	
	std::shared_ptr<Renderer> m_renderer;
	std::unique_ptr<DirectX::Keyboard> m_keyboard;

	HINSTANCE m_hInstance;
	HighResolutionClock m_clock;
};
