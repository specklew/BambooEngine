#include "pch.h"
#include "CommandContext.h"
#include "FrameAccumulationPass.h"

#include "Constants.h"
#include "PassRegisters.h"
#include "RootSignatureLibrary.h"
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
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, ACCUM_REG_CURRENT);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, ACCUM_REG_ACCUM);
    ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, ACCUM_REG_DISPLAY);
    rootParams[0].InitAsDescriptorTable(3, ranges);

    // [1] Root constant for frameCount
    rootParams[1].InitAsConstants(1, ACCUM_REG_CB);

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(
        _countof(rootParams), rootParams,
        0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    m_rootSignature = RootSignatureLibrary::Get().Create(m_device.Get(), rootSigDesc,
                                                         L"FrameAccumulationPass Root Signature");
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

        m_accumulationBuffer = std::make_unique<Texture>(m_device, desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"FrameAccumulation Buffer");
    }

    // Display buffer (R16G16B16A16_FLOAT — HDR, tonemapped by PostProcessPass)
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Alignment = 0;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        m_displayBuffer = std::make_unique<Texture>(m_device, desc,
            D3D12_RESOURCE_STATE_COPY_SOURCE, L"FrameDisplay Buffer");
    }
}

void FrameAccumulationPass::CreatePSO()
{
    spdlog::debug("Creating PSO for accumulation compute shader");

    m_program = ShaderProgramCache::Get().GetOrCreateCompute(m_device.Get(), m_rootSignature.Get(),
        "resources/shaders/accumulation.cs.shader", L"FrameAccumulationPass PSO");
}

void FrameAccumulationPass::Render(Texture& currentFrameOutput)
{
    if (!m_initialized)
        return;

    // Get descriptor handles
    UINT descriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetCPUDescriptorHandleForHeapStart());

    // Create SRV for current frame at slot 0 (t0)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(currentFrameOutput.GetUnderlyingResource().Get(), &srvDesc, cpuHandle);

    // Create UAV for accumulation buffer at slot 1 (u0)
    cpuHandle.Offset(1, descriptorSize);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_accumulationBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc, cpuHandle);

    // Create UAV for display buffer at slot 2 (u1)
    cpuHandle.Offset(1, descriptorSize);
    uavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    m_device->CreateUnorderedAccessView(m_displayBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc, cpuHandle);

    // Bind root signature and PSO
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_program->GetPipelineState());

    // Set descriptor heap and bind descriptor table
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootDescriptorTable(0, m_descriptorHeap->GetGPUDescriptorHandleForHeapStart());

    // Bind root constant for frameCount
    m_commandList->SetComputeRoot32BitConstant(1, m_frameCount + 1, 0);

    // Dispatch
    UINT width = Window::Get().GetWidth();
    UINT height = Window::Get().GetHeight();
    UINT threadsX = (width + 7) / 8;   // 8x8 thread groups
    UINT threadsY = (height + 7) / 8;
    CommandContext::Get().Dispatch(threadsX, threadsY, 1);

    m_frameCount++;
}

void FrameAccumulationPass::Update(double elapsedTime)
{
    m_accumulatedTime += elapsedTime;
}

void FrameAccumulationPass::Reset()
{
    m_frameCount      = 0;
    m_accumulatedTime = 0.0;
    m_resetCount++;
}

void FrameAccumulationPass::OnResize()
{
    if (!m_initialized)
        return;

    spdlog::debug("Resizing frame accumulation buffers");
    m_accumulationBuffer.reset();
    m_displayBuffer.reset();
    CreateResources();
    Reset();
}
