#include "pch.h"

#include "Application.h"

#include "Window.h"
#include "Headless.h"
#include "HeadlessRunner.h"
#include "ResourceManager/ResourceManager.h"

#include <cstdlib>

Application::Application(HINSTANCE hInstance) : m_hInstance(hInstance)
{
	m_renderer = std::make_shared<Renderer>();
}

int Application::Run()
{
	SetupLoggingLevel();

	m_keyboard = std::make_unique<DirectX::Keyboard>();

	int argc = 0;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	const HeadlessArgs headlessArgs = ParseHeadlessArgs(argc, argv);
	LocalFree(argv);

	HeadlessConfig headlessConfig;
	const LONG winW = 1280, winH = 720;
	RECT windowRect = { 0, 0, winW, winH };
	if (headlessArgs.headless)
	{
		headlessConfig = LoadHeadlessConfig("SavedUserData/headless.json");
		windowRect = { 0, 0, static_cast<LONG>(headlessConfig.width), static_cast<LONG>(headlessConfig.height) };
	}

	Window::Create(m_hInstance, windowRect, this, headlessArgs.headless);

	// Benchmarks must not pay the debug layer's uneven per-submit validation
	// tax (see Renderer::g_enableDebugLayer), so headless opts out unless
	// --debug-layer asks for a correctness run instead of a timing one.
	Renderer::g_enableDebugLayer = !headlessArgs.headless || headlessArgs.debugLayer;

	m_renderer->Initialize();
	m_ready = true;

	// Set after Initialize so CVar defaults do not reset them. In a headless run
	// HeadlessRunner arms the dump itself, once per technique after its capture —
	// frame 0 pays for PSO creation and the geometry bake, which would make the
	// node cost table meaningless.
	if (headlessArgs.rdgTimings)
		CVarSystem::Get()->SetCVarInt(StringId("rdg.timings"), 1);
	if (headlessArgs.rdgDump && !headlessArgs.headless)
		CVarSystem::Get()->SetCVarInt(StringId("rdg.dump"), 1);

	// --cvar name=value. Float or int is decided by which one the CVar system
	// already knows the name as, so nothing has to state the type on the command
	// line; an unknown name is a typo and says so rather than being ignored.
	for (const std::string& assignment : headlessArgs.cvarAssignments)
	{
		const size_t separator = assignment.find('=');
		if (separator == std::string::npos)
		{
			spdlog::error("--cvar expects name=value, got '{}'", assignment);
			continue;
		}

		const std::string name  = assignment.substr(0, separator);
		const std::string value = assignment.substr(separator + 1);
		const StringId     id(name.c_str());

		if (CVarSystem::Get()->GetFloatCVar(id))
			CVarSystem::Get()->SetCVarFloat(id, std::stof(value));
		else if (CVarSystem::Get()->GetIntCVar(id))
			CVarSystem::Get()->SetCVarInt(id, std::stoi(value));
		else
		{
			spdlog::error("--cvar '{}' is not a float or int CVar", name);
			continue;
		}
		spdlog::info("--cvar {} = {}", name, value);
	}

	int exitCode = 0;
	if (headlessArgs.headless)
	{
		HeadlessRunner runner(*m_renderer, headlessArgs, headlessConfig);
		exitCode = runner.Run();

		// The engine's global teardown (static ComPtr device, ImGui, resource
		// manager) segfaults on shutdown — harmless under the interactive loop
		// (process is exiting anyway) but it would clobber our exit code. The
		// captures are already flushed to disk, so fast-exit with the real code.
		spdlog::default_logger()->flush();
		std::_Exit(exitCode);
	}

	GameLoop();

	m_ready = false;
	m_renderer->CleanUp();

	ResourceManager::Get().ReleaseResources();
	//ReportLiveObjects();

	return exitCode;
}

void Application::GameLoop()
{
	constexpr double kFpsTitleUpdateInterval = 0.5; // seconds between title refreshes
	double fpsAccumulatedTime = 0.0;
	uint32_t fpsFrameCount = 0;

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

		fpsAccumulatedTime += m_clock.GetDeltaSeconds();
		fpsFrameCount++;
		if (fpsAccumulatedTime >= kFpsTitleUpdateInterval)
		{
			const double fps = fpsFrameCount / fpsAccumulatedTime;
			const double frameMs = 1000.0 * fpsAccumulatedTime / fpsFrameCount;

			wchar_t suffix[64];
			swprintf_s(suffix, L"%.0f FPS (%.2f ms)", fps, frameMs);
			Window::Get().SetTitleSuffix(suffix);

			fpsAccumulatedTime = 0.0;
			fpsFrameCount = 0;
		}
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
 
void Application::OnMouseWheel(int delta)
{
	m_renderer->OnMouseWheel(delta);
}

void Application::OnKeyDown(WPARAM btnState)
{
	m_renderer->OnKeyDown(btnState);
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
