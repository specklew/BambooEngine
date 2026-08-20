#include "pch.h"

#include "Application.h"

// Agility SDK opt-in (ADR 0020 R1). Without these two exports the process runs on
// whatever D3D12 the OS ships, which caps us at DXR tier 1.1 / shader model 6.8 no
// matter which headers we compile against — SER needs tier 1.2 and SM 6.9, which
// exist only in the redistributable runtime. They must be exported from the EXE
// itself and read before the first D3D12 call, which is why they live here.
// D3D12SDKPath is relative to the EXE; the package's targets drop D3D12Core.dll and
// d3d12SDKLayers.dll into that folder. 1.721.3-preview is a PREVIEW runtime: it
// loads only with Windows Developer Mode on, and silently falls back to the in-box
// runtime otherwise — so the startup tier/shader-model log is the check that it took.
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, INT nCmdShow) {

	spdlog::enable_backtrace(32);
	Application app(hInstance);

	return app.Run();
}
