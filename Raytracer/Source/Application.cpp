#include "pch.h"

#include "Application.h"

#include "Window.h"

Application::Application(HINSTANCE hInstance) : m_hInstance(hInstance) {}

void Application::Run()
{
	Window::Create(m_hInstance, { 0, 0, 800, 600 });

	m_renderer.Initialize();

	GameLoop();
}

void Application::GameLoop()
{
	MSG msg = {};
	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		m_clock.Tick();

		m_renderer.Update(m_clock.GetDeltaSeconds(), m_clock.GetTotalSeconds());
		m_renderer.Render(m_clock.GetDeltaSeconds(), m_clock.GetTotalSeconds());
	}
}
