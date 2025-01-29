#include "pch.h"

#include "Renderer.h"

#include <chrono>

#include "Helpers.h"
#include "Window.h"

using namespace Microsoft::WRL;

void Renderer::Initialize()
{
	SetupDevice();
	CreateCommandQueue();
	CreateCommandAllocators();
	CreateFence();
	CreateSwapChain();
	
	CreateCommandList();
	ResetCommandList();
	
	CreateRTVDescriptorHeap();
	CreateRenderTargetViews();
}

void Renderer::Update(double elapsedTime, double totalTime)
{
}

void Renderer::Render(double elapsedTime, double totalTime)
{
	auto allocator = m_d3d12CommandAllocators[m_frameIndex];
	auto backBuffer = m_d3d12RenderTargets[m_frameIndex];

	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
		   backBuffer.Get(),
		   D3D12_RESOURCE_STATE_PRESENT,
		   D3D12_RESOURCE_STATE_RENDER_TARGET);

		m_d3d12CommandList->ResourceBarrier(1, &barrier);

		FLOAT clearColor[4] = { 0.3f, 0.6f, 0.9f, 1.0f };

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
			m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			m_frameIndex,
			m_rtvDescriptorSize);

		m_d3d12CommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	}

	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
			backBuffer.Get(),
			D3D12_RESOURCE_STATE_RENDER_TARGET,
			D3D12_RESOURCE_STATE_PRESENT);

		m_d3d12CommandList->ResourceBarrier(1, &barrier);

		ThrowIfFailed(m_d3d12CommandList->Close());
	}

	ID3D12CommandList* const commandLists[] = { m_d3d12CommandList.Get() };
	
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	UINT presentFlags = CheckTearingSupport() ? DXGI_PRESENT_ALLOW_TEARING : 0;

	ThrowIfFailed(m_dxgiSwapChain->Present(0, presentFlags));

	ResetCommandList();
	
	// Signalling
	m_fenceValues[m_frameIndex]++;
	ThrowIfFailed(m_d3d12CommandQueue->Signal(m_d3d12Fence.Get(), m_fenceValues[m_frameIndex]));

	// Wait for fence value
	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
	if (m_d3d12Fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
	{
		std::chrono::milliseconds duration = std::chrono::milliseconds::max();
		ThrowIfFailed(m_d3d12Fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
		WaitForSingleObject(m_fenceEvent, static_cast<DWORD>(duration.count()));
	}
}	

void Renderer::SetupDevice()
{

#ifdef _DEBUG
	ComPtr<ID3D12Debug> debugController;
	D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
	debugController->EnableDebugLayer();
#endif
	
	UINT createFactoryFlags = 0;

#ifdef _DEBUG
	createFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

	ThrowIfFailed(::CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory)));
	
	m_dxgiAdapter = GetHardwareAdapter();

	if (m_dxgiAdapter)
	{
		m_d3d12Device = GetDeviceForAdapter(m_dxgiAdapter);
	}
}

void Renderer::CreateCommandQueue()
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};

	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_d3d12CommandQueue)));
}

void Renderer::CreateCommandAllocators()
{
	for (UINT i = 0; i < NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_d3d12CommandAllocators[i])));
	}
}

void Renderer::CreateFence()
{
	ThrowIfFailed(m_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_d3d12Fence)));

	m_fenceValues[m_frameIndex]++;

	m_fenceEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);

	if (m_fenceEvent == nullptr)
	{
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}
}

void Renderer::CreateSwapChain()
{
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = NUM_FRAMES;
	swapChainDesc.Width = Window::Get().GetWidth();
	swapChainDesc.Height = Window::Get().GetHeight();
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	
	IDXGISwapChain1* swapChain1;
	
	ThrowIfFailed(m_dxgiFactory->CreateSwapChainForHwnd(
		m_d3d12CommandQueue.Get(),
		Window::Get().GetHandle(),
		&swapChainDesc,
		nullptr,
		nullptr,
		&swapChain1));

	ThrowIfFailed(m_dxgiFactory->MakeWindowAssociation(Window::Get().GetHandle(), DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain1->QueryInterface(IID_PPV_ARGS(&m_dxgiSwapChain)));

	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
}

void Renderer::CreateCommandList()
{
	ThrowIfFailed(m_d3d12Device->CreateCommandList(
		0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_d3d12CommandAllocators[m_frameIndex].Get(),
		nullptr,
		IID_PPV_ARGS(&m_d3d12CommandList)));

	ThrowIfFailed(m_d3d12CommandList->Close());
}

void Renderer::ResetCommandList()
{
	ThrowIfFailed(m_d3d12CommandAllocators[m_frameIndex]->Reset());
	ThrowIfFailed(m_d3d12CommandList->Reset(m_d3d12CommandAllocators[m_frameIndex].Get(), nullptr));
}

void Renderer::CreateRTVDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = NUM_FRAMES;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

	ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_d3d12RTVDescriptorHeap)));

	m_rtvDescriptorSize = m_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void Renderer::CreateRenderTargetViews()
{
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	for (UINT i = 0; i < NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		m_d3d12Device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);

		rtvHandle.ptr += m_rtvDescriptorSize;
	}
}

bool Renderer::CheckTearingSupport()
{
	BOOL tearingAllowed;

	if (FAILED(m_dxgiFactory->CheckFeatureSupport(
		DXGI_FEATURE_PRESENT_ALLOW_TEARING,
		&tearingAllowed,
		sizeof(tearingAllowed))))
	{
		tearingAllowed = false;
	}
	

	return tearingAllowed;
}

ComPtr<IDXGIAdapter4> Renderer::GetHardwareAdapter(bool useWarp)
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

ComPtr<ID3D12Device2> Renderer::GetDeviceForAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter)
{
	ComPtr<ID3D12Device2> device;
	ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

#ifdef _DEBUG

	ComPtr<ID3D12InfoQueue> pInfoQueue;

	if (SUCCEEDED(device.As(&pInfoQueue)))
	{
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

		D3D12_MESSAGE_SEVERITY severities[] = {
			D3D12_MESSAGE_SEVERITY_INFO
		};

		D3D12_MESSAGE_ID denyIds[] = {
			D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
			D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
			D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE
		};

		D3D12_INFO_QUEUE_FILTER newFilter = {};
		newFilter.DenyList.NumSeverities = _countof(severities);
		newFilter.DenyList.pSeverityList = severities;
		newFilter.DenyList.NumIDs = _countof(denyIds);
		newFilter.DenyList.pIDList = denyIds;

		ThrowIfFailed(pInfoQueue->PushStorageFilter(&newFilter));
	}

#endif

	return device;
}