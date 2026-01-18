#include "pch.h"

#include "Renderer.h"

#include <algorithm>

#include "AccelerationStructures.h"
#include "Camera.h"
#include "Utils/CVars.h"
#include "SceneResources/GameObject.h"
#include "imgui.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

#include "InputElements.h"
#include "SceneResources/ModelLoading.h"
#include "SceneResources/Primitive.h"
#include "RaytracePass.h"
#include "SceneResources/Scene.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Window.h"
#include "Resources/ConstantBuffer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/Texture.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/Material.h"
#include "SceneResources/Model.h"
#include "tinygltf/tiny_gltf.h"

#ifdef _DEBUG
#define ENABLE_GPU_BASED_VALIDATION 1
#endif

using namespace Microsoft::WRL;

static AutoCVarEnum g_currentScene("renderer.initialScene", "Specifies which scene to load on startup.", ModelLoading::LOADED_SCENES::A_BEAUTIFUL_GAME);
static AutoCVarFloat g_cameraSpeed("renderer.camera.speed", "Specifies the base speed of camera", 1.0f, CVarFlags::EditDrag, 1.0f, 20.0f);

void Renderer::Initialize()
{
	spdlog::info("Initializing renderer...");

	m_camera = std::make_shared<Camera>();
	m_keyboardTracker = std::make_shared<DirectX::Keyboard::KeyboardStateTracker>();
	
	SetupDeviceAndDebug();
	CheckTearingSupport();
	
	if (!CheckRayTracingSupport()) 	throw std::runtime_error("Raytracing is not supported on this device.");;
	
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

	// Execute depth stencil creation commands
	ThrowIfFailed(m_d3d12CommandList->Close());
	
	ID3D12CommandList* commandLists[] = {m_d3d12CommandList.Get()};
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	
	FlushCommandQueue();
	ResetCommandList();
	// Finish execution - reset command list for next setup commands

	CreateDescriptorHeaps();
	CreateWorldProjCBV();

	m_loadedScenes = ModelLoading::LoadAllScenes(*this);

	m_material = std::make_shared<Material>();
	m_material->m_data.albedoColor = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	
	CreateRasterizationRootSignature();

	CreatePipelineState();

	m_d3d12CommandList->Close();
	commandLists[0] = m_d3d12CommandList.Get();
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	
	FlushCommandQueue();
	ResetCommandList();
	
	SetupAccelerationStructures();

	m_raytracePass = std::make_shared<RaytracePass>();
	m_raytracePass->Initialize(g_d3d12Device, m_d3d12CommandList, m_accelerationStructures, m_vertexBuffer, m_indexBuffer, m_srvCbvUavDescriptorHeap);

	spdlog::info("Renderer initialized successfully.");

	InitializeImGui();
	
	FlushCommandQueue();
	ResetCommandList();
}
float speedMultiplier = 1.0f;
void Renderer::Update(double elapsedTime, double totalTime)
{
	auto key_state = DirectX::Keyboard::Get().GetState();

	if (key_state.LeftShift || key_state.RightShift)
	{
		if (speedMultiplier < 100.0f) speedMultiplier *= 1.002f;
	}
	else
	{
		speedMultiplier = 1.0f;
	}
	
	if (key_state.W)
	{
		m_camera->AddPosition(m_camera->GetForward() * static_cast<float>(elapsedTime) * g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.S)
	{
		m_camera->AddPosition(m_camera->GetForward() * static_cast<float>(elapsedTime) * -g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.D)
	{
		m_camera->AddPosition(m_camera->GetRight() * static_cast<float>(elapsedTime) * g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.A)
	{
		m_camera->AddPosition(m_camera->GetRight() * static_cast<float>(elapsedTime) * -g_cameraSpeed.Get() * speedMultiplier);
	}

	if (key_state.F2)
	{
		OnShaderReload();
	}
	
	using namespace DirectX;

    //SimpleMath::Matrix world = m_world;
	SimpleMath::Matrix viewProjection = XMLoadFloat4x4(&m_camera->GetViewProjectionMatrix());
	SimpleMath::Matrix view = XMLoadFloat4x4(&m_camera->GetViewMatrix());
	SimpleMath::Matrix viewProj = XMLoadFloat4x4(&m_camera->GetViewProjectionMatrix());
	SimpleMath::Matrix projection = XMLoadFloat4x4(&m_camera->GetProjectionMatrix());
	
	XMVECTOR det;

	struct ObjectConstants
	{
		XMFLOAT4X4 ViewProj = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 View = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 Projection = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 ViewInverse = MathUtils::XMFloat4x4Identity();
		XMFLOAT4X4 ProjectionInverse = MathUtils::XMFloat4x4Identity();
	};
	
	ObjectConstants constants;
	XMStoreFloat4x4(&constants.ViewProj, XMMatrixTranspose(viewProjection));
	XMStoreFloat4x4(&constants.View, XMMatrixTranspose(view));
	XMStoreFloat4x4(&constants.Projection, XMMatrixTranspose(projection));
	XMStoreFloat4x4(&constants.ViewInverse, XMMatrixInverse(&det, view));
	XMStoreFloat4x4(&constants.ProjectionInverse, XMMatrixInverse(&det, projection));
	
	memcpy(&m_mappedData[0], &constants, sizeof(constants));

	m_raytracePass->Update(view, viewProj);

	if (previousScene != g_currentScene.Get())
	{
		m_raytracePass->OnSceneChange(g_currentScene.Get());
		previousScene = g_currentScene.Get();

		ID3D12CommandList* commandLists[] = {m_d3d12CommandList.Get()};
		m_d3d12CommandList->Close();
		commandLists[0] = m_d3d12CommandList.Get();
		m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

		FlushCommandQueue();
		ResetCommandList();
	}
}

void Renderer::Render(double elapsedTime, double totalTime)
{
	RenderImGui();
	
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

	ID3D12DescriptorHeap* descriptorHeaps[] = {m_srvCbvUavDescriptorHeap.Get()};

	m_d3d12CommandList->SetPipelineState(m_pipelineStateObject.Get());
	m_d3d12CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	m_d3d12CommandList->SetGraphicsRootSignature(m_rootSignature.Get());


	if (m_rasterize)
	{
		for (const auto& go : m_loadedScenes[g_currentScene.Get()]->GetGameObjects())
		{
			auto gpuAddress = go->m_worldMatrixBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
			m_d3d12CommandList->SetGraphicsRootConstantBufferView(1, gpuAddress);
			
			for (const auto& primitive : go->GetModel()->GetMeshes())
			{
				gpuAddress = primitive->m_material->m_materialBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
				m_d3d12CommandList->SetGraphicsRootConstantBufferView(2, gpuAddress);
				
				auto vertexBuffer = primitive->GetVertexBuffer();
				auto indexBuffer = primitive->GetIndexBuffer();
	
				m_d3d12CommandList->IASetVertexBuffers(0, 1, &vertexBuffer->GetVertexBufferView());
				m_d3d12CommandList->IASetIndexBuffer(&indexBuffer->GetIndexBufferView());
				m_d3d12CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

				m_d3d12CommandList->SetGraphicsRootDescriptorTable(0, m_srvCbvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
				
				// Assume first index and vertex are 0 ( buffers are only for one object )
				m_d3d12CommandList->DrawIndexedInstanced(indexBuffer->GetIndexCount(), 1, 0, 0, 0);
			}
		}
	}
	else
	{
		m_raytracePass->Render(backBuffer);
	}

	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_d3d12CommandList.Get());
	
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
			backBuffer.Get(),
			D3D12_RESOURCE_STATE_RENDER_TARGET,
			D3D12_RESOURCE_STATE_PRESENT);

		m_d3d12CommandList->ResourceBarrier(1, &barrier);
		
	}
	
	ID3D12CommandList* const commandLists[] = { m_d3d12CommandList.Get() };
	
	ThrowIfFailed(m_d3d12CommandList->Close());
	
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	UINT presentFlags = m_tearingSupport ? DXGI_PRESENT_ALLOW_TEARING : 0; // TODO: do not check every time
	
	ThrowIfFailed(m_dxgiSwapChain->Present(0, presentFlags));

	FlushCommandQueue();
	ResetCommandList();
}

void Renderer::CleanUp()
{
	FlushCommandQueue();

	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
	
	m_projectionMatrixConstantBuffer->GetUnderlyingResource()->Unmap(0, nullptr);
}

void Renderer::OnResize()
{
	assert(g_d3d12Device && "Attempted to resize window without device.");
	assert(m_dxgiSwapChain && "Attempted to resize window without swap chain.");
	
	auto& window = Window::Get();

	FlushCommandQueue();

	for (int i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		m_d3d12RenderTargets[i].Reset();
	}

	int swapChainFlags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	ThrowIfFailed(m_dxgiSwapChain->ResizeBuffers(
		Constants::Graphics::NUM_FRAMES,
		window.GetWidth(),
		window.GetHeight(),
		m_backBufferFormat,
		swapChainFlags));

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		g_d3d12Device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);
		rtvHandle.Offset(1, m_rtvDescriptorSize);
	}
	
	m_depthStencilBuffer.Reset();

	CreateDepthStencilView();
	SetScissorRect();
	SetViewport();
	
	m_raytracePass->OnResize();

	m_d3d12CommandList->Close();
	ID3D12CommandList* commandLists[] = {m_d3d12CommandList.Get()};
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	
	FlushCommandQueue();
	ResetCommandList();
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

		m_camera->AddRotationEuler(DirectX::SimpleMath::Vector3(dy, -dx, 0.0f));
		
		// Restrict the angle mPhi.
		m_phi = std::clamp(m_phi, 0.01f - DirectX::XM_PIDIV2,  DirectX::XM_PIDIV2 - 0.01f);
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

void Renderer::OnKeyDown(unsigned long long btnState) const
{
	const auto state = DirectX::Keyboard::Get().GetState();
	m_keyboardTracker->Update(state);
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
		g_d3d12Device = GetDeviceForAdapter(dxgiAdapter);
	}
}

void Renderer::CreateCommandQueue()
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};

	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(g_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_d3d12CommandQueue)));
}

void Renderer::CreateCommandAllocators()
{
	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(g_d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_d3d12CommandAllocators[i])));
	}
}

void Renderer::CreateFence()
{
	ThrowIfFailed(g_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_d3d12Fence)));

	m_fenceValue++;

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
	ThrowIfFailed(g_d3d12Device->CreateCommandList(
		0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_d3d12CommandAllocators[m_frameIndex].Get(),
		nullptr,
		IID_PPV_ARGS(&m_d3d12CommandList)));

	ThrowIfFailed(m_d3d12CommandList->Close());
}

void Renderer::ResetCommandList() const
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

	ThrowIfFailed(g_d3d12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_d3d12RTVDescriptorHeap)));

	m_rtvDescriptorSize = g_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void Renderer::CreateRenderTargetViews()
{
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_d3d12RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	for (UINT i = 0; i < Constants::Graphics::NUM_FRAMES; ++i)
	{
		ThrowIfFailed(m_dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_d3d12RenderTargets[i])));
		g_d3d12Device->CreateRenderTargetView(m_d3d12RenderTargets[i].Get(), nullptr, rtvHandle);

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
	
	ThrowIfFailed(g_d3d12Device->CreateCommittedResource(
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
	
	g_d3d12Device->CreateDepthStencilView(
		m_depthStencilBuffer.Get(),
		&viewDesc,
		m_d3d12DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	
	m_d3d12CommandList->ResourceBarrier(
		1,
		&CD3DX12_RESOURCE_BARRIER::Transition(
			m_depthStencilBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_DEPTH_WRITE));
}

void Renderer::CreateDSVDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.NumDescriptors = 1;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.NodeMask = 0;

	ThrowIfFailed(g_d3d12Device->CreateDescriptorHeap(
		&desc,
		IID_PPV_ARGS(&m_d3d12DSVDescriptorHeap)));
}

void Renderer::CreateDescriptorHeaps()
{
	///	|							|								|							|							|
	///	|	SRV IMGUI TEXTURE (1)	|	UAV RAYTRACING OUTPUT (1)	|	SRV TLAS (MAX_SCENES)	|	TEXTURES (MAX_TEXTURE)	|
	///	|							|								|							|							|
	
	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	desc.NumDescriptors = Constants::Graphics::NUM_BASE_DESCRIPTORS + Constants::Engine::MAX_SCENES + Constants::Graphics::MAX_TEXTURES;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	desc.NodeMask = 0;

	ThrowIfFailed(g_d3d12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_srvCbvUavDescriptorHeap)));
}

void Renderer::CreateWorldProjCBV()
{
	ComPtr<ID3D12Resource> cbvUav;
	
	g_d3d12Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(256 * 5),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&cbvUav));

	m_projectionMatrixConstantBuffer = std::make_shared<ConstantBuffer>(g_d3d12Device, cbvUav);
	m_projectionMatrixConstantBuffer->SetResourceName(L"Camera and World Proj CBV Resource");
	
	ThrowIfFailed(m_projectionMatrixConstantBuffer->GetUnderlyingResource()->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedData)));
	
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
	cbvDesc.BufferLocation = m_projectionMatrixConstantBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
	cbvDesc.SizeInBytes = Align(256ULL * 5, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

	auto descriptorDesc = m_srvCbvUavDescriptorHeap->GetDesc();
	auto increment = g_d3d12Device->GetDescriptorHandleIncrementSize(descriptorDesc.Type);

	D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
	cbvHandle.ptr = m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + increment;
	
	g_d3d12Device->CreateConstantBufferView(&cbvDesc, cbvHandle);
}

void Renderer::CreateRasterizationRootSignature()
{
	constexpr int num_params = 3;
	
	CD3DX12_ROOT_PARAMETER rootParameters[num_params];
	
	D3D12_DESCRIPTOR_RANGE cbvRange;
	cbvRange.BaseShaderRegister = 0;
	cbvRange.NumDescriptors = 1;
	cbvRange.RegisterSpace = 0;
	cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
	cbvRange.OffsetInDescriptorsFromTableStart = 1;

	D3D12_DESCRIPTOR_RANGE rtRange;
	rtRange.BaseShaderRegister = 0;
	rtRange.NumDescriptors = 1;
	rtRange.RegisterSpace = 0;
	rtRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	rtRange.OffsetInDescriptorsFromTableStart = 2;

	D3D12_DESCRIPTOR_RANGE tlasRange;
	tlasRange.BaseShaderRegister = 0;
	tlasRange.NumDescriptors = 1;
	tlasRange.RegisterSpace = 0;
	tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	tlasRange.OffsetInDescriptorsFromTableStart = 3;

	D3D12_DESCRIPTOR_RANGE textureRange;
	textureRange.BaseShaderRegister = 1;
	textureRange.NumDescriptors = Constants::Graphics::MAX_TEXTURES;
	textureRange.RegisterSpace = 0;
	textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	textureRange.OffsetInDescriptorsFromTableStart = 4;

	D3D12_DESCRIPTOR_RANGE ranges[] = {cbvRange, rtRange, tlasRange, textureRange};
	
	rootParameters[0].InitAsDescriptorTable(4, ranges);
	rootParameters[1].InitAsConstantBufferView(1); // Model index buffer
	rootParameters[2].InitAsConstantBufferView(2);

	auto static_samplers = GetStaticSamplers();
	
	CD3DX12_ROOT_SIGNATURE_DESC desc = {};
	desc.NumParameters = num_params;
	desc.pParameters = rootParameters;
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
	desc.NumStaticSamplers = static_samplers.size();
	desc.pStaticSamplers = static_samplers.data();

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
	
	ThrowIfFailed(g_d3d12Device->CreateRootSignature(
		0,
		serializedRootSignature->GetBufferPointer(),
		serializedRootSignature->GetBufferSize(),
		IID_PPV_ARGS(&m_rootSignature)));
}

void Renderer::CreatePipelineState()
{
	auto& rm = ResourceManager::Get();
	
	auto psh = rm.GetOrLoadShader(AssetId("resources/shaders/colorShader.ps.shader"));
	m_pixelShader = rm.shaders.GetResource(psh).bytecode;
	auto vsh = rm.GetOrLoadShader(AssetId("resources/shaders/colorShader.vs.shader"));
	m_vertexShader = rm.shaders.GetResource(vsh).bytecode;
	
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

	ComPtr<ID3D12PipelineState> pso;
	ThrowIfFailed(g_d3d12Device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)));
	pso->SetName(L"Default Pipeline State");
	m_pipelineStateObject = pso;
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
	m_fenceValue++;
	ThrowIfFailed(m_d3d12CommandQueue->Signal(m_d3d12Fence.Get(), m_fenceValue));

	if (m_d3d12Fence->GetCompletedValue() < m_fenceValue)
	{
		ThrowIfFailed(m_d3d12Fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent));
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}

	m_frameIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();
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

bool Renderer::CheckRayTracingSupport() const
{
	spdlog::info("Checking ray tracing support...");
	
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
	ThrowIfFailed(g_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
	if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
	{
		spdlog::error("Ray tracing not supported!");
		return false;
	}

	spdlog::info("Ray tracing supported!");
	return true;
}

void Renderer::ExtractBLASesFromSceneModels(const Scene& scene, AccelerationStructures& accelerationStructures)
{
	for (auto model : scene.GetModels())
	{
		std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vertex_buffers;
		std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> index_buffers;
		
		for (auto prim : model->GetMeshes())
		{

			std::pair<ComPtr<ID3D12Resource>, uint32_t> vertex_pair = {
				prim->GetVertexBuffer()->GetUnderlyingResource(),
				prim->GetVertexBuffer()->GetVertexCount()
			};
			
			std::pair<ComPtr<ID3D12Resource>, uint32_t> index_pair = {
				prim->GetIndexBuffer()->GetUnderlyingResource(),
				prim->GetIndexBuffer()->GetIndexCount()
			};
			
			vertex_buffers.emplace_back(vertex_pair);
			index_buffers.emplace_back(index_pair);
		}

		AccelerationStructureBuffers bottomLevelBuffers = accelerationStructures.CreateBottomLevelAS(
			g_d3d12Device.Get(),
			m_d3d12CommandList.Get(),
			vertex_buffers,
			index_buffers);

		m_modelsBLASes.emplace(model, std::make_shared<AccelerationStructureBuffers>(bottomLevelBuffers));
	}
}

void Renderer::CreateTLASForScene(const Scene& scene, AccelerationStructures& accelerationStructures)
{
	for (auto go : scene.GetGameObjects())
	{
		auto model = go->GetModel();
		auto blasIt = m_modelsBLASes.find(model);
		if (blasIt != m_modelsBLASes.end())
		{
			DirectX::XMMATRIX worldMatrix = DirectX::XMLoadFloat4x4(&go->m_worldMatrix);
			auto instance = std::pair(blasIt->second->p_result, worldMatrix);
			accelerationStructures.GetInstances().emplace_back(instance);
		}
		else
		{
			spdlog::warn("Model for GameObject not found in BLASes map during TLAS setup.");
		}
	}

	accelerationStructures.CreateTopLevelAS(g_d3d12Device.Get(), m_d3d12CommandList.Get(), accelerationStructures.GetInstances(), false);
}

void Renderer::SetupAccelerationStructures()
{
	spdlog::debug("Setting up acceleration structures");
	
	for (auto scene : m_loadedScenes)
	{
		spdlog::debug("Setting up AS for scene: {}", scene->GetName());
		auto accelerationStructures = std::make_shared<AccelerationStructures>();
		ExtractBLASesFromSceneModels(*scene, *accelerationStructures);
		CreateTLASForScene(*scene, *accelerationStructures);
		m_accelerationStructures.emplace_back(accelerationStructures);
	}

	spdlog::debug("Executing command list with BLAS generation");
	m_d3d12CommandList->Close();
	ID3D12CommandList* commandLists[] = {m_d3d12CommandList.Get()};
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	FlushCommandQueue();
}

void Renderer::CreateTextureSRV(const std::shared_ptr<Texture>& texture)
{
	assert(texture && "Passed texture cannot be null!");
	assert(texture->GetUnderlyingResource() && "Texture resources cannot be null!");

	spdlog::debug("Setting up texture SRV");

	const auto& resource = texture->GetUnderlyingResource();
	const auto& desc = resource->GetDesc();
	
	D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
	if (desc.DepthOrArraySize == 1)
	{
		srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srv_desc.Texture2D.MostDetailedMip = 0;
		srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
		srv_desc.Texture2D.MipLevels = desc.MipLevels;
	}
	else
	{
		spdlog::error("Texture 2D array functionality is not yet supported!");
		// TODO: In case that texture is a TEXTURE 2D ARRAY
	}
	srv_desc.Format = desc.Format;
	srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; // TODO: Check what this does and why default is best here.

	auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	handle.Offset(4 + texture->GetTextureIndex(),  g_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));

	g_d3d12Device->CreateShaderResourceView(resource.Get(), &srv_desc, handle);
}

void Renderer::InitializeImGui()
{
	// Make process DPI aware and get main monitor scale
	ImGui_ImplWin32_EnableDpiAwareness();
	float mainScaleImGui = ImGui_ImplWin32_GetDpiScaleForMonitor(::MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY));
	
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking

	ImGui::StyleColorsDark();

	ImGuiStyle& style = ImGui::GetStyle();
	style.ScaleAllSizes(mainScaleImGui);

	ImGui_ImplWin32_Init(Window::Get().GetHandle());
	
	ImGui_ImplDX12_InitInfo init_info = {};
	init_info.Device = g_d3d12Device.Get();
	init_info.CommandQueue = m_d3d12CommandQueue.Get();
	init_info.NumFramesInFlight = Constants::Graphics::NUM_FRAMES;
	init_info.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	init_info.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	
	init_info.LegacySingleSrvCpuDescriptor.ptr = (m_srvCbvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr);
	init_info.LegacySingleSrvGpuDescriptor.ptr = (m_srvCbvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr);
	init_info.SrvDescriptorHeap = m_srvCbvUavDescriptorHeap.Get();
	
	ImGui_ImplDX12_Init(&init_info);
	spdlog::info("ImGui initialized successfully.");
}

void Renderer::RenderImGui()
{
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
	
	CVarSystem::Get()->DrawImguiEditor();

	ImGui::Render();
}

void Renderer::OnShaderReload()
{
	spdlog::info("Reloading shaders...");
	ResourceManager::Get().RecompileAllShaders();
	
	ThrowIfFailed(m_d3d12CommandList->Close());
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

	FlushCommandQueue();
	
	spdlog::info("Creating pipeline state for new shaders...");
	CreatePipelineState();
	m_raytracePass->OnShaderReload();

	FlushCommandQueue();
	ResetCommandList();
	spdlog::info("Shaders reloaded successfully.");
}

std::array<const CD3DX12_STATIC_SAMPLER_DESC, Constants::Graphics::STATIC_SAMPLERS_COUNT> Renderer::GetStaticSamplers()
{
	// Apps usually only need a handful of samplers. So just define them
	// all up front and keep them available as part of the root signature.

	const CD3DX12_STATIC_SAMPLER_DESC pointWrap(
		0, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
		1, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC linearWrap(
		2, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC linearClamp(
		3, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
	const CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(
		4, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressW
		0.0f, // mipLODBias
		8); // maxAnisotropy
	const CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(
		5, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressW
		0.0f, // mipLODBias
		8); // maxAnisotropy
	;

	return { pointWrap, pointClamp, linearWrap, linearClamp, anisotropicWrap, anisotropicClamp };
}

void Renderer::ToggleRasterization()
{
	m_rasterize = !m_rasterize;
}

std::shared_ptr<Primitive> Renderer::CreatePrimitive(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::shared_ptr<Material>& material)
{
	ComPtr<ID3D12Resource> vertex_upload_buffer;
	ComPtr<ID3D12Resource> index_upload_buffer;

	auto cpuVertex = static_cast<BYTE*>(malloc(vertices.size() * sizeof(Vertex)));
	auto cpuIndex = static_cast<BYTE*>(malloc(indices.size() * sizeof(uint32_t)));
	memcpy(cpuVertex, vertices.data(), vertices.size() * sizeof(Vertex));
	memcpy(cpuIndex, indices.data(), indices.size() * sizeof(uint32_t));
	
	auto vertex_buffer_resource = RenderingUtils::CreateDefaultBuffer(g_d3d12Device.Get(), m_d3d12CommandList.Get(), cpuVertex, vertices.size() * sizeof(Vertex), vertex_upload_buffer);
	auto index_buffer_resource = RenderingUtils::CreateDefaultBuffer(g_d3d12Device.Get(), m_d3d12CommandList.Get(), cpuIndex, indices.size() * sizeof(uint32_t), index_upload_buffer);
	
	// Need to be closed and executed to create buffers before the upload buffers go out of scope.
	m_d3d12CommandList->Close();
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	FlushCommandQueue();
	// WHY is resetting the allocator impossible if:
	// 1. commands are closed
	// 2. commands are executed
	// 3. command queue is flushed
	// D3D12 ERROR: ID3D12CommandAllocator::Reset: The command allocator cannot be reset because a command list is currently being recorded with the allocator. [ EXECUTION ERROR #543: COMMAND_ALLOCATOR_CANNOT_RESET]
	// Is it possible that in the meantime something gets recorded to the command list?
	// ---
	// We have 3 allocators for each of the triple buffered frames
	// When we reset, we reset only the current frame's allocator
	// But maybe we haven't yet presented the frame, and we need to??? is that it?

	// After all it seems that the FlushGPU method was not functioning correctly, I never quite researched why that was the case. But it seems that there was an issue with the fence value.
	// There was a unique fence value for each allocator (frame) and somehow it was not being updated properly. TODO: Check out why was that for the next iteration of the engine.
	ResetCommandList();

	auto vertex_buffer = std::make_shared<VertexBuffer>(g_d3d12Device, vertex_buffer_resource, static_cast<UINT>(vertices.size()), sizeof(Vertex));
	auto index_buffer = std::make_shared<IndexBuffer>(g_d3d12Device, index_buffer_resource, static_cast<UINT>(indices.size()), DXGI_FORMAT_R32_UINT);

	auto primitive = std::make_shared<Primitive>(vertex_buffer, index_buffer, material);

	AssertFreeClear(&cpuVertex);
	AssertFreeClear(&cpuIndex);
	
	return primitive;
}

std::shared_ptr<Texture> Renderer::CreateTextureFromGLTF(const tinygltf::Image& image)
{
	ComPtr<ID3D12Resource> upload_buffer;

	auto texture_resource = RenderingUtils::CreateDefaultTexture(g_d3d12Device.Get(), m_d3d12CommandList.Get(), image, upload_buffer);

	std::shared_ptr<Texture> texture = std::make_shared<Texture>(g_d3d12Device, texture_resource);
	CreateTextureSRV(texture);

	m_d3d12CommandList->Close();
	ID3D12CommandList* commandLists[] = { m_d3d12CommandList.Get() };
	m_d3d12CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	FlushCommandQueue();
	ResetCommandList();

	m_textures.push_back(texture);
	
	return texture;
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
	
	ThrowIfFailed(g_d3d12Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&uploadBuffer)));

	uploadBuffer->SetName(L"IntermediateUploadBuffer");
	ComPtr<ID3D12Resource> defaultBuffer;

	ThrowIfFailed(g_d3d12Device->CreateCommittedResource(
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

std::shared_ptr<GameObject> Renderer::InstantiateGameObject()
{
	ComPtr<ID3D12Resource> buffer;
	
	g_d3d12Device->CreateCommittedResource(
	&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
	D3D12_HEAP_FLAG_NONE,
	&CD3DX12_RESOURCE_DESC::Buffer(256),
	D3D12_RESOURCE_STATE_GENERIC_READ,
	nullptr,
	IID_PPV_ARGS(&buffer));

	auto constantBuffer = std::make_shared<ConstantBuffer>(g_d3d12Device, buffer);
	constantBuffer->SetResourceName(L"Model CBV Resource");

	auto game_object = std::make_shared<GameObject>();
	game_object->m_worldMatrixBuffer = constantBuffer;

	auto matrix = DirectX::XMMatrixIdentity();
	DirectX::XMFLOAT4X4 modelWorldMatrix;
	DirectX::XMStoreFloat4x4(&modelWorldMatrix, matrix);
	game_object->UpdateWorldMatrix(modelWorldMatrix);

	return game_object;
}
