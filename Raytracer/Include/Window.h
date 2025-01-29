#pragma once

class Window
{
public:
	static HRESULT Create(HINSTANCE hInstance, RECT windowRect);
	static Window& Get();

	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;

	long GetWidth() const;
	long GetHeight() const;
	HWND GetHandle() const;
	
private:
	Window() = default;
	~Window();

	HRESULT Initialize();

	static Window* instance;

	HINSTANCE m_hInstance;
	HWND m_windowHandle;
	RECT m_windowRect;
};