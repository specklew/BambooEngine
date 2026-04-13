#include "pch.h"
#include "PostProcessPass.h"

#include "Constants.h"
#include "Shader.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"

void PostProcessPass::Initialize(
    Microsoft::WRL::ComPtr<ID3D12Device5> device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    spdlog::info("Initializing post-process pass...");

    m_device = device;
    m_commandList = commandList;

    // Create descriptor heap for SRV + UAV
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 2;  // 1 SRV + 1 UAV
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_descriptorHeap)));

    CreateResources();
    CreateRootSignature();
    CreatePSO();

    m_initialized = true;
    spdlog::info("Post-process pass initialized successfully.");
}

void PostProcessPass::CreateRootSignature()
{
    spdlog::debug("Creating root signature for post-process pass");

    CD3DX12_ROOT_PARAMETER rootParams[2];
    CD3DX12_DESCRIPTOR_RANGE ranges[2];

    // [0] Descriptor table with SRV and UAV
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // SRV at t0
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // UAV at u0
    rootParams[0].InitAsDescriptorTable(2, ranges);

    // [1] Root constant for exposure (b0)
    rootParams[1].InitAsConstants(1, 0);

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(
        _countof(rootParams), rootParams,
        0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));

    m_rootSignature->SetName(L"PostProcessPass Root Signature");
}

void PostProcessPass::CreateResources()
{
    spdlog::debug("Creating post-process output buffer");

    UINT width = Window::Get().GetWidth();
    UINT height = Window::Get().GetHeight();

    // Output buffer (R8G8B8A8_UNORM)
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
            IID_PPV_ARGS(&m_outputBuffer)));

        m_outputBuffer->SetName(L"PostProcess Output Buffer");
    }
}

void PostProcessPass::CreatePSO()
{
    spdlog::debug("Creating PSO for post-process compute shader");

    auto& rm = ResourceManager::Get();
    auto shaderHandle = rm.GetOrLoadShader(AssetId("resources/shaders/postprocess.cs.shader"));
    m_computeShaderBlob = rm.shaders.GetResource(shaderHandle).bytecode;

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = m_rootSignature.Get();
    desc.CS = CD3DX12_SHADER_BYTECODE(m_computeShaderBlob->GetBufferPointer(), m_computeShaderBlob->GetBufferSize());
    desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_pso)));
    m_pso->SetName(L"PostProcessPass PSO");
}

void PostProcessPass::Render(
    const Microsoft::WRL::ComPtr<ID3D12Resource>& input,
    const Microsoft::WRL::ComPtr<ID3D12Resource>& backBuffer,
    float exposure)
{
    if (!m_initialized)
        return;

    // Get descriptor handles
    UINT descriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetCPUDescriptorHandleForHeapStart());

    // Create SRV for input at slot 0 (t0)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(input.Get(), &srvDesc, cpuHandle);

    // Create UAV for output buffer at slot 1 (u0)
    cpuHandle.Offset(1, descriptorSize);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputBuffer.Get(), nullptr, &uavDesc, cpuHandle);

    // Bind root signature and PSO
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_pso.Get());

    // Set descriptor heap and bind descriptor table
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootDescriptorTable(0, m_descriptorHeap->GetGPUDescriptorHandleForHeapStart());

    // Bind exposure as root constant (float bits cast to uint32)
    uint32_t exposureBits;
    memcpy(&exposureBits, &exposure, sizeof(float));
    m_commandList->SetComputeRoot32BitConstant(1, exposureBits, 0);

    // Transition input to SRV
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            input.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m_commandList->ResourceBarrier(1, &transition);
    }

    // Transition output buffer to UAV
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputBuffer.Get(),
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
        CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_outputBuffer.Get());
        m_commandList->ResourceBarrier(1, &uavBarrier);
    }

    // Transition output buffer back to copy source
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &transition);
    }

    // Transition input back to COPY_SOURCE
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            input.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_commandList->ResourceBarrier(1, &transition);
    }

    // Copy output buffer to back buffer
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            backBuffer.Get(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_DEST);
        m_commandList->ResourceBarrier(1, &transition);
    }

    m_commandList->CopyResource(backBuffer.Get(), m_outputBuffer.Get());

    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            backBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        m_commandList->ResourceBarrier(1, &transition);
    }
}

void PostProcessPass::OnResize()
{
    if (!m_initialized)
        return;

    spdlog::debug("Resizing post-process output buffer");
    m_outputBuffer.Reset();
    CreateResources();
}
