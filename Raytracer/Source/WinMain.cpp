#include "pch.h"

#include "Application.h"

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, INT nCmdShow) {

	spdlog::enable_backtrace(32);
	Application app(hInstance);

	app.Run();

	return 0;
}
