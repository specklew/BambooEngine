#include "pch.h"
#include "Window.h"
#include "Constants.h"

static LRESULT CALLBACK WndProc(
	HWND hWnd,
	UINT message,
	WPARAM wParam, LPARAM lParam);

Window* Window::instance = nullptr;

HRESULT Window::Create(HINSTANCE hInstance, RECT windowRect)
{
	if (instance != nullptr)
	{
		return E_FAIL;
	}

	instance = new Window();

	instance->m_hInstance = hInstance;
	instance->m_windowRect = windowRect;

	return instance->Initialize();
}

Window& Window::Get()
{
	return *instance;
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

	WCHAR windowClassName[MAX_STRING_LEN];
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

LRESULT WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}
