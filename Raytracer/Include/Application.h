#pragma once

#include "HighResolutionClock.h"
#include "Renderer.h"

class Application
{
public:
	Application(HINSTANCE hInstance);

	void Run();

private:
	friend class Window;
	
	void GameLoop();
	void OnMouseMove(WPARAM btnState, int x, int y);
	void OnKeyDown(WPARAM btnState);
	void ReportLiveObjects();
	void SetupLoggingLevel();
	
	std::shared_ptr<Renderer> m_renderer;

	HINSTANCE m_hInstance;
	HighResolutionClock m_clock;
};
