#include "pch.h"

#include "Application.h"

#include "Window.h"
#include "ResourceManager/ResourceManager.h"

Application::Application(HINSTANCE hInstance) : m_hInstance(hInstance)
{
	m_renderer = std::make_shared<Renderer>();
}

void Application::Run()
{
	SetupLoggingLevel();

	m_keyboard = std::make_unique<DirectX::Keyboard>();
	
	Window::Create(m_hInstance, { 0, 0, 800, 600 }, this);
	
	m_renderer->Initialize();
	m_ready = true;

	GameLoop();

	m_ready = false;
	m_renderer->CleanUp();
	m_renderer.reset();

	ResourceManager::Get().ReleaseResources();
	//ReportLiveObjects();
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

		m_renderer->Update(m_clock.GetDeltaSeconds(), m_clock.GetTotalSeconds());
		m_renderer->Render(m_clock.GetDeltaSeconds(), m_clock.GetTotalSeconds());
	}
}

void Application::OnResize()
{
	m_renderer->OnResize();
}

void Application::OnMouseMove(WPARAM btnState, int x, int y)
{
	m_renderer->OnMouseMove(btnState, x, y);
}

void Application::OnKeyDown(WPARAM btnState)
{
	if (btnState == VK_SPACE)
	{
		m_renderer->ToggleRasterization();
	}
	else
	{
		m_renderer->OnKeyDown(btnState);
	}
}

void Application::ReportLiveObjects()
{
#ifdef _DEBUG
	spdlog::info("***** Reporting live objects *****");
	Microsoft::WRL::ComPtr<IDXGIDebug1> dxgiDebug;
	if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug))))
	{
		ThrowIfFailed(dxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL)));
	}

	SPDLOG_INFO("***** Done Reporting Live Objects *****");
#endif
}

void Application::SetupLoggingLevel()
{
#ifdef _DEBUG
	spdlog::set_level(spdlog::level::debug);
#endif
}
