#include "pch.h"
#include "FrameAccumulationPass.h"

#include "Constants.h"
#include "Shader.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"

void FrameAccumulationPass::Initialize(
    Microsoft::WRL::ComPtr<ID3D12Device5> device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    spdlog::info("Initializing frame accumulation pass...");

    m_device = device;
    m_commandList = commandList;

    // Create descriptor heap for SRV + UAVs
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 3;  // 1 SRV + 2 UAVs
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_descriptorHeap)));

    CreateResources();
    CreateRootSignature();
    CreatePSO();

    m_initialized = true;
    spdlog::info("Frame accumulation pass initialized successfully.");
}

void FrameAccumulationPass::CreateRootSignature()
{
    spdlog::debug("Creating root signature for accumulation pass");

    CD3DX12_ROOT_PARAMETER rootParams[2];
    CD3DX12_DESCRIPTOR_RANGE ranges[3];

    // [0] Descriptor table with all resources
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // SRV at t0
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // UAV at u0
    ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // UAV at u1
    rootParams[0].InitAsDescriptorTable(3, ranges);

    // [1] Root constant for frameCount (b0)
    rootParams[1].InitAsConstants(1, 0);

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(
        _countof(rootParams), rootParams,
        0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));

    m_rootSignature->SetName(L"FrameAccumulationPass Root Signature");
}

void FrameAccumulationPass::CreateResources()
{
    spdlog::debug("Creating frame accumulation buffers");

    UINT width = Window::Get().GetWidth();
    UINT height = Window::Get().GetHeight();

    // Accumulation buffer (R32G32B32A32_FLOAT)
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Alignment = 0;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_accumulationBuffer)));

        m_accumulationBuffer->SetName(L"FrameAccumulation Buffer");
    }

    // Display buffer (R8G8B8A8_UNORM)
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Alignment = 0;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            nullptr,
            IID_PPV_ARGS(&m_displayBuffer)));

        m_displayBuffer->SetName(L"FrameDisplay Buffer");
    }
}

void FrameAccumulationPass::CreatePSO()
{
    spdlog::debug("Creating PSO for accumulation compute shader");

    auto& rm = ResourceManager::Get();
    auto shaderHandle = rm.GetOrLoadShader(AssetId("resources/shaders/accumulation.cs.shader"));
    m_computeShaderBlob = rm.shaders.GetResource(shaderHandle).bytecode;

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = m_rootSignature.Get();
    desc.CS = CD3DX12_SHADER_BYTECODE(m_computeShaderBlob->GetBufferPointer(), m_computeShaderBlob->GetBufferSize());
    desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_pso)));
    m_pso->SetName(L"FrameAccumulationPass PSO");
}

void FrameAccumulationPass::Render(
    const Microsoft::WRL::ComPtr<ID3D12Resource>& currentFrameOutput)
{
    if (!m_initialized)
        return;

    // Get descriptor handles
    UINT descriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetCPUDescriptorHandleForHeapStart());

    // Create SRV for current frame at slot 0 (t0)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(currentFrameOutput.Get(), &srvDesc, cpuHandle);

    // Create UAV for accumulation buffer at slot 1 (u0)
    cpuHandle.Offset(1, descriptorSize);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_accumulationBuffer.Get(), nullptr, &uavDesc, cpuHandle);

    // Create UAV for display buffer at slot 2 (u1)
    cpuHandle.Offset(1, descriptorSize);
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    m_device->CreateUnorderedAccessView(m_displayBuffer.Get(), nullptr, &uavDesc, cpuHandle);

    // Bind root signature and PSO
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_pso.Get());

    // Set descriptor heap and bind descriptor table
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootDescriptorTable(0, m_descriptorHeap->GetGPUDescriptorHandleForHeapStart());

    // Bind root constant for frameCount
    m_commandList->SetComputeRoot32BitConstant(1, m_frameCount + 1, 0);

    // Transition display buffer to UAV
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_displayBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &transition);
    }

    // Dispatch
    UINT width = Window::Get().GetWidth();
    UINT height = Window::Get().GetHeight();
    UINT threadsX = (width + 7) / 8;   // 8x8 thread groups
    UINT threadsY = (height + 7) / 8;
    m_commandList->Dispatch(threadsX, threadsY, 1);

    // UAV barrier to flush writes
    {
        CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_displayBuffer.Get());
        m_commandList->ResourceBarrier(1, &uavBarrier);
    }

    // Transition display buffer back to copy source
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_displayBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &transition);
    }

    m_frameCount++;
}

void FrameAccumulationPass::Update(double elapsedTime)
{
    m_accumulatedTime += elapsedTime;
}

void FrameAccumulationPass::Reset()
{
    m_frameCount = 0;
    m_accumulatedTime = 0.0;
    m_pendingScreenshots = 0;
}

void FrameAccumulationPass::OnResize()
{
    if (!m_initialized)
        return;

    spdlog::debug("Resizing frame accumulation buffers");
    m_accumulationBuffer.Reset();
    m_displayBuffer.Reset();
    CreateResources();
    Reset();
}
