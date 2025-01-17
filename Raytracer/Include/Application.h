#pragma once
#include <chrono>

#include "HighResolutionClock.h"
#include "Renderer.h"

class Application
{
public:
	Application(HINSTANCE hInstance);

	void Run();

private:
	void GameLoop();

	Renderer m_renderer;

	HINSTANCE m_hInstance;
	HighResolutionClock m_clock;
};
