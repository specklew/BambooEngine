#include "pch.h"
#include "GraphicsDevice.h"

#include "Utils/Utils.h"

using namespace Microsoft::WRL;

#ifdef _DEBUG
#define ENABLE_GPU_BASED_VALIDATION 1
// Routes every D3D12 debug-layer message to spdlog so validation errors/warnings
// appear in the engine console (and the headless log), not just the attached
// debugger's output window.
static void CALLBACK D3D12DebugMessageCallback(
	D3D12_MESSAGE_CATEGORY /*category*/, D3D12_MESSAGE_SEVERITY severity,
	D3D12_MESSAGE_ID /*id*/, LPCSTR description, void* /*context*/)
{
	switch (severity)
	{
	case D3D12_MESSAGE_SEVERITY_CORRUPTION:
	case D3D12_MESSAGE_SEVERITY_ERROR:
		spdlog::error("[D3D12] {}", description);
		break;
	case D3D12_MESSAGE_SEVERITY_WARNING:
		spdlog::warn("[D3D12] {}", description);
		break;
	default:
		spdlog::debug("[D3D12] {}", description);
		break;
	}
}
#endif

void GraphicsDevice::Initialize(bool enableDebugLayer)
{
	m_enableDebugLayer = enableDebugLayer;

#ifdef _DEBUG
	ComPtr<ID3D12Debug> debugController;
	D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
	if (m_enableDebugLayer)
		debugController->EnableDebugLayer();

	// DRED: on device-removed, ThrowIfFailed dumps auto-breadcrumbs (which
	// command in which command list hung/faulted) + page-fault allocation info.
	// Breadcrumbs inject per-command marker writes driver-side, so gate on the
	// same switch as the debug layer — headless benchmark runs stay untaxed.
	if (m_enableDebugLayer)
	{
		ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSettings;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dredSettings))))
		{
			dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
		}
	}
#endif

	UINT createFactoryFlags = 0;

#ifdef _DEBUG
	createFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

	ThrowIfFailed(::CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory)));

#ifdef _DEBUG
	if (m_enableDebugLayer) {
	ComPtr<ID3D12Debug1> debugController1;
	ThrowIfFailed(debugController.As(&debugController1));

	debugController1->SetEnableGPUBasedValidation(ENABLE_GPU_BASED_VALIDATION);
	}
#endif

	if (ComPtr<IDXGIAdapter4> dxgiAdapter = GetHardwareAdapter())
	{
		m_device = GetDeviceForAdapter(dxgiAdapter);
	}
}

void GraphicsDevice::CreateCommandQueue()
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};

	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));
}

void GraphicsDevice::CreateCommandAllocators()
{
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[i])));
	}
}

void GraphicsDevice::CreateFence()
{
	ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));

	m_fenceValue++;

	m_fenceEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);

	if (m_fenceEvent == nullptr)
	{
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}
}

void GraphicsDevice::CreateSwapChain(HWND windowHandle, uint32_t width, uint32_t height, DXGI_FORMAT backBufferFormat)
{
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = Constants::Graphics::NUM_FRAMES;
	swapChainDesc.Width = width;
	swapChainDesc.Height = height;
	swapChainDesc.Format = backBufferFormat;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

	IDXGISwapChain1* swapChain1;

	ThrowIfFailed(m_dxgiFactory->CreateSwapChainForHwnd(
		m_commandQueue.Get(),
		windowHandle,
		&swapChainDesc,
		nullptr,
		nullptr,
		&swapChain1));

	ThrowIfFailed(m_dxgiFactory->MakeWindowAssociation(windowHandle, DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain1->QueryInterface(IID_PPV_ARGS(&m_swapChain)));

	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}

bool GraphicsDevice::CheckTearingSupport()
{
	BOOL tearingAllowed;

	if (FAILED(m_dxgiFactory->CheckFeatureSupport(
		DXGI_FEATURE_PRESENT_ALLOW_TEARING,
		&tearingAllowed,
		sizeof(tearingAllowed))))
	{
		tearingAllowed = false;
	}

	m_tearingSupport = tearingAllowed;
	return tearingAllowed;
}

bool GraphicsDevice::CheckRayTracingSupport() const
{
	spdlog::info("Checking ray tracing support...");

	D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
	ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
	if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1)
	{
		spdlog::error("Ray tracing not supported!");
		return false;
	}

	// VXPG fingerprint / cvis visibility kernels use inline RayQuery (Tier 1.1,
	// above) plus [WaveSize(32)] ballot packing, which needs shader model 6.6.
	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_6 };
	ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)));
	if (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_6)
	{
		spdlog::error("Shader model 6.6 not supported (required by VXPG wave-intrinsic passes)!");
		return false;
	}

	// VXPG light tree: uint64 bitonic sort keys need Int64ShaderOps; the
	// byte-identical uint16 TreeNode layout needs native 16-bit shader ops.
	D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
	ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)));
	if (!options1.Int64ShaderOps)
	{
		spdlog::error("Int64 shader ops not supported (required by the VXPG light-tree sort)!");
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4 = {};
	ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4)));
	if (!options4.Native16BitShaderOpsSupported)
	{
		spdlog::error("Native 16-bit shader ops not supported (required by the VXPG light-tree nodes)!");
		return false;
	}

	// VXPG guided integrator carries its sampling pdfs in double, faithful to
	// SIByL (ADR 0003 integrator-swap section).
	D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
	ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));
	if (!options.DoublePrecisionFloatShaderOps)
	{
		spdlog::error("Double-precision shader ops not supported (required by the VXPG guided integrator pdfs)!");
		return false;
	}

	spdlog::info("Ray tracing supported!");
	return true;
}

void GraphicsDevice::FlushCommandQueue()
{
	m_fenceValue++;
	ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), m_fenceValue));

	if (m_fence->GetCompletedValue() < m_fenceValue)
	{
		ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent));
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}

	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}

void GraphicsDevice::RefreshFrameIndex()
{
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}

ComPtr<IDXGIAdapter4> GraphicsDevice::GetHardwareAdapter(bool useWarp)
{
	ComPtr<IDXGIAdapter1> adapter1;
	ComPtr<IDXGIAdapter4> adapter;

	if (useWarp)
	{
		ThrowIfFailed(m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter1)));
		ThrowIfFailed(adapter1.As(&adapter));
		return adapter;
	}

	size_t maxDedicatedVideoMemory = 0;

	for (UINT i = 0; m_dxgiFactory->EnumAdapters1(i, &adapter1) != DXGI_ERROR_NOT_FOUND; ++i){
		DXGI_ADAPTER_DESC1 desc;
		adapter1->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			continue;

		if (SUCCEEDED(D3D12CreateDevice(adapter1.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
		{
			if (desc.DedicatedVideoMemory > maxDedicatedVideoMemory)
			{
				maxDedicatedVideoMemory = desc.DedicatedVideoMemory;
				ThrowIfFailed(adapter1.As(&adapter));
			}
		}
	}

	return adapter;
}

ComPtr<ID3D12Device5> GraphicsDevice::GetDeviceForAdapter(ComPtr<IDXGIAdapter1> adapter)
{
	ComPtr<ID3D12Device5> device;
	ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

#ifdef _DEBUG
	if (m_enableDebugLayer) { // InfoQueue interfaces only exist with the debug layer
	ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_infoQueue)));
	// Only break into the debugger on genuine memory corruption; errors and
	// warnings are logged (below) so headless runs surface them instead of
	// aborting on a breakpoint with no debugger attached.
	ThrowIfFailed(m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true));
	m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, false);
	m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, false);

	// Mirror every debug-layer message into spdlog (console + headless log).
	ComPtr<ID3D12InfoQueue1> infoQueue1;
	if (SUCCEEDED(m_infoQueue.As(&infoQueue1)))
	{
		infoQueue1->RegisterMessageCallback(&D3D12DebugMessageCallback,
			D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &m_debugMessageCallbackCookie);
	}
	else
	{
		spdlog::warn("ID3D12InfoQueue1 unavailable; D3D12 messages will only reach the debugger output.");
	}
	}
#endif


	return device;
}
