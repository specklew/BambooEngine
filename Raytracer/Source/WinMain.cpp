#include "pch.h"

#include "Application.h"

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, INT nCmdShow) {

	Application app(hInstance);

	app.Run();

	return 0;
}
