#include "pch.h"
#include "Window.h"

#include "Application.h"
#include "Constants.h"
#include "imgui.h"
#include <Keyboard.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam))
		return true;
	
	return Window::Get().MsgProc(hWnd, message, wParam, lParam);
}

Window* Window::instance = nullptr;

HRESULT Window::Create(HINSTANCE hInstance, RECT windowRect, Application* app)
{
	if (instance != nullptr)
	{
		return E_FAIL;
	}

	instance = new Window();

	instance->application = app;
	instance->m_hInstance = hInstance;
	instance->m_windowRect = windowRect;

	return instance->Initialize();
}

Window& Window::Get()
{
	return *instance;
}

long Window::GetWidth() const
{
	return m_windowRect.right - m_windowRect.left;
}

long Window::GetHeight() const
{
	return m_windowRect.bottom - m_windowRect.top;
}

HWND Window::GetHandle() const
{
	return m_windowHandle;
}

Window::~Window()
{
	delete instance;
}

HRESULT Window::Initialize()
{
	std::wstring windowTitle = L"Raytracer";

	int argc;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	//ParseCommandLineArgs(argv, argc); TODO: Parse

	WCHAR windowClassName[Constants::MAX_STRING_LEN];
	wcscpy_s(windowClassName, windowTitle.c_str());

	WNDCLASSEX windowClass = { 0 };
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
	windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	windowClass.hIcon = LoadIcon(0, IDI_APPLICATION);
	windowClass.hIconSm = LoadIcon(0, IDI_APPLICATION);
	windowClass.lpszClassName = windowClassName;
	windowClass.lpszMenuName = nullptr;
	windowClass.hInstance = m_hInstance;
	windowClass.lpfnWndProc = &WndProc;

	RegisterClassEx(&windowClass);

	HWND handle = CreateWindow(
		windowClassName,
		windowTitle.c_str(),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		m_windowRect.right - m_windowRect.left,
		m_windowRect.bottom - m_windowRect.top,
		nullptr,
		nullptr,
		m_hInstance,
		nullptr);

	m_windowHandle = handle;

	ShowWindow(handle, SW_SHOW);

	return S_OK;
}

LRESULT Window::MsgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) const
{
	using DirectX::Keyboard;
	
	switch (message)
	{
	case WM_ACTIVATE:
		Keyboard::ProcessMessage(message, wParam, lParam);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_MOUSEMOVE:
		application->OnMouseMove(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		break;
	case WM_KEYDOWN:
		if (wParam == VK_RETURN && (lParam & 0x60000000) == 0x20000000)
		{
			// This is where you'd implement the classic ALT+ENTER hotkey for fullscreen toggle
			// But I'm not going to... heh
		}
		Keyboard::ProcessMessage(message, wParam, lParam);
		application->OnKeyDown(wParam);
		break;
	case WM_KEYUP:
		Keyboard::ProcessMessage(message, wParam, lParam);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}
