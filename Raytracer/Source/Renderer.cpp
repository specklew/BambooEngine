#include "pch.h"

#include "Renderer.h"

#include <algorithm>
#include <chrono>

#include "AccelerationStructures.h"
#include "imgui.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

#include "Helpers.h"
#include "InputElements.h"
#include "RaytracePass.h"
#include "ResourceManager.h"
#include "Shader.h"
#include "Window.h"

#define ENABLE_GPU_BASED_VALIDATION 1

using namespace Microsoft::WRL;

Vertex vertices[] = {
	{ DirectX::XMFLOAT3(-1.0f, -1.0f, -1.0f), DirectX::XMFLOAT4(0, 0, 0, 1) },
	{ DirectX::XMFLOAT3(-1.0f, +1.0f, -1.0f), DirectX::XMFLOAT4(0, 1, 0, 1) },
	{ DirectX::XMFLOAT3(+1.0f, +1.0f, -1.0f), DirectX::XMFLOAT4(1, 1, 0, 1) },
	{ DirectX::XMFLOAT3(+1.0f, -1.0f, -1.0f), DirectX::XMFLOAT4(1, 0, 0, 1) },
	{ DirectX::XMFLOAT3(-1.0f, -1.0f, +1.0f), DirectX::XMFLOAT4(0, 0, 1, 1) },
	{ DirectX::XMFLOAT3(-1.0f, +1.0f, +1.0f), DirectX::XMFLOAT4(0, 1, 1, 1) },
	{ DirectX::XMFLOAT3(+1.0f, +1.0f, +1.0f), DirectX::XMFLOAT4(1, 1, 1, 1) },
	{ DirectX::XMFLOAT3(+1.0f, -1.0f, +1.0f), DirectX::XMFLOAT4(1, 0, 1, 1) }
};

std::uint32_t indices[] = {
	// front face
	0, 1, 2,
	0, 2, 3,
	// back face
	4, 6, 5,
	4, 7, 6,
	// left face
	4, 5, 1,
	4, 1, 0,
	// right face
	3, 2, 6,
	3, 6, 7,
	// top face
	1, 5, 6,
	1, 6, 2,
	// bottom face
	4, 0, 3,
	4, 3, 7
};

struct ObjectConstants
{
	DirectX::XMFLOAT4X4 WorldViewProj = Math::Identity4x4();
};

constexpr UINT64 vbByteSize = 8 * sizeof(Vertex);

auto& resourceManager = ResourceManager::Get();

void Renderer::Initialize()
{

	auto psh = resourceManager.GetOrLoadShader(AssetId("resources/shaders/colorShader.ps.shader"));
	m_pixelShader = resourceManager.shaders.GetResource(psh).bytecode;
	auto vsh = resourceManager.GetOrLoadShader(AssetId("resources/shaders/colorShader.vs.shader"));
	m_vertexShader = resourceManager.shaders.GetResource(vsh).bytecode;

	
	SetupDeviceAndDebug();
	CheckTearingSupport();
	CheckRayTracingSupport();
	
	CreateCommandQueue();
	CreateCommandAllocators();
	CreateFence();
	CreateSwapChain();
	
	CreateCommandList();
	ResetCommandList();
	
	CreateRTVDescriptorHeap();
	CreateRenderTargetViews();

	CreateDSVDescriptorHeap();
	CreateDepthStencilView();
	ResetCommandList();
	
	CreateVertexAndIndexBuffer();
	CreateConstantBufferView();
	CreateRootSignature();

	CreatePipelineState();

	m_d3d12CommandList->Close();
	ID3D12CommandList* const commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	
	FlushCommandQueue();
	ResetCommandList();
	
	SetupAccelerationStructures();

	m_raytracePass = std::make_shared<RaytracePass>();
	m_raytracePass->Initialize(m_d3d12Device, m_d3d12CommandList, m_accelerationStructures);
}

void Renderer::Update(double elapsedTime, double totalTime)
{
	using namespace DirectX;

	XMMATRIX P = XMMatrixPerspectiveFovLH(0.25* 3.14159265358979323846f, static_cast<float>(Window::Get().GetWidth()) / static_cast<float>(Window::Get().GetHeight()), 0.1f, 100.0f);
	XMStoreFloat4x4(&m_proj, P);
	
    // Convert Spherical to Cartesian coordinates.
    float x = m_radius*sinf(m_phi)*cosf(m_theta);
    float z = m_radius*sinf(m_phi)*sinf(m_theta);
    float y = m_radius*cosf(m_phi);

    // Build the view matrix.
    XMVECTOR pos = XMVectorSet(x, y, z, 1.0f);
    XMVECTOR target = XMVectorZero();
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX view = XMMatrixLookAtLH(pos, target, up);
    XMStoreFloat4x4(&m_view, view);

    XMMATRIX world = XMLoadFloat4x4(&m_world);
    XMMATRIX proj = XMLoadFloat4x4(&m_proj);
    XMMATRIX worldViewProj = world*view*proj;

	ObjectConstants constants;
	XMStoreFloat4x4(&constants.WorldViewProj, XMMatrixTranspose(worldViewProj));

	m_worldViewProj = XMMatrixTranspose(worldViewProj);
	
	memcpy(&m_mappedData[0], &constants, sizeof(constants));

	m_raytracePass->Update(view, proj);
}

void Renderer::Render(double elapsedTime, double totalTime)
{
	ResetCommandList();
	SetViewport();
	SetScissorRect();
	
	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
	
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
		m_d3d12CommandList->ClearDepthStencilView(
			m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			D3D12_CLEAR_FLAG_DEPTH,
			1.0f,
			0,
			0,
			nullptr);
		
		m_d3d12CommandList->OMSetRenderTargets(1,
		&rtvHandle,
		true,
		&m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	}

	ID3D12DescriptorHeap* descriptorHeaps[] = {m_CBVDescriptorHeap.Get()};

	m_d3d12CommandList->SetPipelineState(m_pipelineStateObject.Get());
	m_d3d12CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	m_d3d12CommandList->SetGraphicsRootSignature(m_rootSignature.Get());

	if (m_rasterize)
	{
		m_d3d12CommandList->IASetVertexBuffers(0, 1, m_vertexBuffers);
    	m_d3d12CommandList->IASetIndexBuffer(&m_indexBufferView);
    	m_d3d12CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    	m_d3d12CommandList->SetGraphicsRootDescriptorTable(0, m_CBVDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    	
    	m_d3d12CommandList->DrawIndexedInstanced(_countof(indices), 1, 0, 0, 0);
	}
	else
	{
		m_accelerationStructures->CreateTopLevelAS(m_d3d12Device, m_d3d12CommandList, m_accelerationStructures->GetInstances(), true);
		m_raytracePass->Render(backBuffer);
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
	UINT presentFlags = m_tearingSupport ? DXGI_PRESENT_ALLOW_TEARING : 0; // TODO: do not check every time

	ThrowIfFailed(m_dxgiSwapChain->Present(0, presentFlags));

	FlushCommandQueue();
}

void Renderer::CleanUp()
{
	FlushCommandQueue();
	
	m_constantBuffer->Unmap(0, nullptr);
}

void Renderer::OnMouseMove(unsigned long long btnState, int x, int y)
{
	if((btnState & MK_LBUTTON) != 0)
	{
		// Make each pixel correspond to a quarter of a degree.
		float dx = DirectX::XMConvertToRadians(0.25f*static_cast<float> (x - m_lastMousePosX));
		float dy = DirectX::XMConvertToRadians(0.25f*static_cast<float> (y - m_lastMousePosY));
		// Update angles based on input to orbit camera around box.
		m_theta += dx;
		m_phi += dy;
		// Restrict the angle mPhi.
		m_phi = std::clamp(m_phi, 0.1f,  3.1415f - 0.1f);
	}
	else if((btnState & MK_RBUTTON) != 0)
	{
		// Make each pixel correspond to 0.005 unit in the scene.
		float dx = 0.005f*static_cast<float>(x - m_lastMousePosX);
		float dy = 0.005f*static_cast<float>(y - m_lastMousePosY);
		// Update the camera radius based on input.
		m_radius += dx - dy;
		// Restrict the radius.
		m_radius = std::clamp(m_radius, 3.0f, 15.0f);
	}
	m_lastMousePosX = x;
	m_lastMousePosY = y;
}

void Renderer::SetupDeviceAndDebug()
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

#ifdef _DEBUG
	ComPtr<ID3D12Debug1> debugController1;
	ThrowIfFailed(debugController.As(&debugController1));

	debugController1->SetEnableGPUBasedValidation(ENABLE_GPU_BASED_VALIDATION);
#endif

	if (ComPtr<IDXGIAdapter4> dxgiAdapter = GetHardwareAdapter())
	{
		m_d3d12Device = GetDeviceForAdapter(dxgiAdapter);
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
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
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
	swapChainDesc.BufferCount = Constants::Graphics::NUM_FRAMES;
	swapChainDesc.Width = Window::Get().GetWidth();
	swapChainDesc.Height = Window::Get().GetHeight();
	swapChainDesc.Format = m_backBufferFormat;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	
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
	ThrowIfFailed(m_d3d12CommandList->Reset(m_d3d12CommandAllocators[m_frameIndex].Get(), m_pipelineStateObject.Get()));
}

void Renderer::CreateRTVDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = Constants::Graphics::NUM_FRAMES;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

	ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_d3d12RTVDescriptorHeap)));

	m_rtvDescriptorSize = m_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void Renderer::CreateRenderTargetViews()
{
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		m_d3d12Device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);

		rtvHandle.ptr += m_rtvDescriptorSize;
	}
}

void Renderer::CreateDepthStencilView()
{
	D3D12_RESOURCE_DESC desc;
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Alignment = 0;
	desc.Width = Window::Get().GetWidth();
	desc.Height = Window::Get().GetHeight();
	desc.MipLevels = 1;
	desc.Format = m_depthStencilFormat;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
	desc.DepthOrArraySize = 1;

	D3D12_CLEAR_VALUE clearValue;
	clearValue.Format = m_depthStencilFormat;
	clearValue.DepthStencil.Depth = 1.0f;
	clearValue.DepthStencil.Stencil = 0.0f;
	
	CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_DEFAULT);
	
	ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
		&heapProp,
		D3D12_HEAP_FLAG_NONE,
		&desc,
		D3D12_RESOURCE_STATE_COMMON,
		&clearValue,
		IID_PPV_ARGS(&m_depthStencilBuffer)));

	D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc;
	viewDesc.Flags = D3D12_DSV_FLAG_NONE;
	viewDesc.Format = m_depthStencilFormat;
	viewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	viewDesc.Texture2D.MipSlice = 0;
	
	m_d3d12Device->CreateDepthStencilView(
		m_depthStencilBuffer.Get(),
		&viewDesc,
		m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	
	m_d3d12CommandList->ResourceBarrier(
		1,
		&CD3DX12_RESOURCE_BARRIER::Transition(
			m_depthStencilBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_DEPTH_WRITE));

	ThrowIfFailed(m_d3d12CommandList->Close());
	
	ID3D12CommandList* commandLists[] = {m_d3d12CommandList.Get()};
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	
	FlushCommandQueue();
}

void Renderer::CreateDSVDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.NumDescriptors = 1;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.NodeMask = 0;

	ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(
		&desc,
		IID_PPV_ARGS(&m_d3d12DSVDescriptorHeap)));
}

void Renderer::CreateVertexAndIndexBuffer()
{
	m_vertexBuffer = CreateDefaultBuffer(vertices, vbByteSize, m_vertexBufferUploader);
	m_vertexBuffer->SetName(L"VertexBuffer");

	m_vertexBufferView = {};
	m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
	m_vertexBufferView.SizeInBytes = vbByteSize;
	m_vertexBufferView.StrideInBytes = sizeof(Vertex);
	
	m_vertexBuffers[0] = m_vertexBufferView;

	m_indexBuffer = CreateDefaultBuffer(indices, sizeof(indices), m_indexBufferUploader);
	m_indexBuffer->SetName(L"IndexBuffer");

	m_indexBufferView = {};
	m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
	m_indexBufferView.SizeInBytes = sizeof(indices);
	m_indexBufferView.Format = DXGI_FORMAT_R32_UINT;
}

void Renderer::CreateConstantBufferView()
{

	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	desc.NumDescriptors = 1;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	desc.NodeMask = 0;

	ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_CBVDescriptorHeap)));
	
	m_d3d12Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(256),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_constantBuffer));

	m_constantBuffer->SetName(L"WorldProjectionBuffer");
	ThrowIfFailed(m_constantBuffer->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedData)));
	
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
	cbvDesc.BufferLocation = m_constantBuffer->GetGPUVirtualAddress();
	cbvDesc.SizeInBytes = static_cast<UINT>(sizeof(DirectX::XMFLOAT4X4) + 255) & ~255;

	m_d3d12Device->CreateConstantBufferView(&cbvDesc, m_CBVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
}

void Renderer::CreateRootSignature()
{
	CD3DX12_ROOT_PARAMETER rootParameters[1];
	CD3DX12_DESCRIPTOR_RANGE cbvTable;

	cbvTable.Init(
		D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
		1,
		0);

	rootParameters[0].InitAsDescriptorTable(1, &cbvTable);

	CD3DX12_ROOT_SIGNATURE_DESC desc = {};
	desc.NumParameters = 1;
	desc.pParameters = rootParameters;
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

	ComPtr<ID3DBlob> serializedRootSignature = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;

	HRESULT hr = D3D12SerializeRootSignature(
		&desc,
		D3D_ROOT_SIGNATURE_VERSION_1,
		&serializedRootSignature,
		&errorBlob);

	if(errorBlob != nullptr)
	{
		OutputDebugStringA(static_cast<char*>(errorBlob->GetBufferPointer()));
	}
	ThrowIfFailed(hr);
	
	ThrowIfFailed(m_d3d12Device->CreateRootSignature(
		0,
		serializedRootSignature->GetBufferPointer(),
		serializedRootSignature->GetBufferSize(),
		IID_PPV_ARGS(&m_rootSignature)));
}

void Renderer::CreatePipelineState()
{
	
	D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
	desc.VS = {static_cast<BYTE*>(m_vertexShader->GetBufferPointer()), m_vertexShader->GetBufferSize()};
	desc.PS = {static_cast<BYTE*>(m_pixelShader->GetBufferPointer()), m_pixelShader->GetBufferSize()};
	desc.InputLayout = {inputLayout, _countof(inputLayout)};
	desc.pRootSignature = m_rootSignature.Get();

	desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	
	desc.SampleMask = UINT_MAX;

	desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	
	desc.NumRenderTargets = 1;
	desc.RTVFormats[0] = m_backBufferFormat;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.DSVFormat = m_depthStencilFormat;

	ThrowIfFailed(m_d3d12Device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_pipelineStateObject)));
	m_pipelineStateObject->SetName(L"Default Pipeline State");
}

void Renderer::SetViewport()
{
	D3D12_VIEWPORT viewport;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Height = Window::Get().GetHeight();
	viewport.Width = Window::Get().GetWidth();
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	m_d3d12CommandList->RSSetViewports(1, &viewport);
}

void Renderer::SetScissorRect()
{
	D3D12_RECT scissorRect;
	scissorRect.left = 0;
	scissorRect.top = 0;
	scissorRect.right = Window::Get().GetWidth();
	scissorRect.bottom = Window::Get().GetHeight();

	m_d3d12CommandList->RSSetScissorRects(1, &scissorRect);
}

void Renderer::FlushCommandQueue()
{
	ThrowIfFailed(m_d3d12CommandQueue->Signal(m_d3d12Fence.Get(), m_fenceValues[m_frameIndex]));
	ThrowIfFailed(m_d3d12Fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));

	WaitForSingleObject(m_fenceEvent, INFINITE);

	m_fenceValues[m_frameIndex]++;
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
	
	m_tearingSupport = tearingAllowed;
	return tearingAllowed;
}

bool Renderer::CheckRayTracingSupport()
{
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
	ThrowIfFailed(m_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
	if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
	{
		throw std::runtime_error("Raytracing is not supported on this device.");
	}	
}

void Renderer::SetupAccelerationStructures()
{
	spdlog::debug("Setting up acceleration structures");
	m_accelerationStructures = std::make_shared<AccelerationStructures>();

	AccelerationStructureBuffers bottomLevelBuffers = m_accelerationStructures->CreateBottomLevelAS(
		m_d3d12Device.Get(), m_d3d12CommandList.Get(), {{m_vertexBuffer, std::size(vertices)}}, {{m_indexBuffer, std::size(indices)}});

	auto instance = std::pair(bottomLevelBuffers.p_result, DirectX::XMMatrixIdentity());
	m_accelerationStructures->GetInstances().push_back(instance);
	m_accelerationStructures->CreateTopLevelAS(
		m_d3d12Device.Get(), m_d3d12CommandList.Get(), m_accelerationStructures->GetInstances(), false); // TODO: handle the instances properly

	spdlog::debug("Executing command list with BLAS generation");
	m_d3d12CommandList->Close();
	ID3D12CommandList* commandLists[] = {m_d3d12CommandList.Get()};
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	FlushCommandQueue();

	m_bottomLevelAS = bottomLevelBuffers.p_result;	// TODO: Store BLAS in the acceleration structure
}

void Renderer::ToggleRasterization()
{
	m_rasterize = !m_rasterize;
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

ComPtr<ID3D12Device5> Renderer::GetDeviceForAdapter(ComPtr<IDXGIAdapter1> adapter)
{
	ComPtr<ID3D12Device5> device;
	ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

#ifdef _DEBUG
	ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_infoQueue)));
	ThrowIfFailed(m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true));
	ThrowIfFailed(m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true));
	ThrowIfFailed(m_infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true));
#endif


	return device;
}

ComPtr<ID3D12Resource> Renderer::CreateDefaultBuffer(const void* initData, UINT64 byteSize, ComPtr<ID3D12Resource> &uploadBuffer)
{
	
	ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&uploadBuffer)));

	uploadBuffer->SetName(L"IntermediateUploadBuffer");
	ComPtr<ID3D12Resource> defaultBuffer;

	ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&defaultBuffer)));

	D3D12_SUBRESOURCE_DATA subResourceData;
	subResourceData.pData = initData;
	subResourceData.RowPitch = byteSize;
	subResourceData.SlicePitch = byteSize;

	m_d3d12CommandList->ResourceBarrier(
		1,
		&CD3DX12_RESOURCE_BARRIER::Transition(
			defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_COPY_DEST));

	UpdateSubresources(m_d3d12CommandList.Get(),
		defaultBuffer.Get(), uploadBuffer.Get(),
		0, 0, 1,
		&subResourceData);

	m_d3d12CommandList->ResourceBarrier(
		1,
		&CD3DX12_RESOURCE_BARRIER::Transition(
			defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_GENERIC_READ));
	
	return defaultBuffer;
}
