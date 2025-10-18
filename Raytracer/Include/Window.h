#pragma once

class Application;

class Window
{
public:
	static HRESULT Create(HINSTANCE hInstance, RECT windowRect, Application* app);
	static Window& Get();

	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;

	long GetWidth() const;
	long GetHeight() const;
	HWND GetHandle() const;

	LRESULT CALLBACK MsgProc(
		HWND hWnd,
		UINT message,
		WPARAM wParam, LPARAM lParam);
	
private:
	Window() = default;
	~Window();

	HRESULT Initialize();
	
	static Window* instance;

	Application* application;
	HINSTANCE m_hInstance;
	HWND m_windowHandle;
	RECT m_windowRect;
};
