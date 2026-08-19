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

    CreateResources();
    CreateRootSignature();
    CreatePSO();

    m_initialized = true;
    spdlog::info("Frame accumulation pass initialized successfully.");
}

// Current frame in, running average and display copy out. Two tables because the
// input is an SRV and the outputs are UAVs, and each is its own NUM_FRAMES-deep
// block of the global heap: offsets here are relative to the frame slot's base,
// which Render() hands to SetTable.
static constexpr BindingSlot kAccumInputSlots[] = {
    PassTableEntryAt("gCurrent", BindingKind::Srv, ACCUM_REG_CURRENT, 0, 1, /*tableIndex*/ 0),
};
static constexpr BindingSlot kAccumTargetSlots[] = {
    PassTableEntryAt("gAccum", BindingKind::Uav, ACCUM_REG_ACCUM, 0, 1, /*tableIndex*/ 1),
    PassTableEntryAt("gDisplay", BindingKind::Uav, ACCUM_REG_DISPLAY, 1, 1, /*tableIndex*/ 1),
};
// frameCount, varianceEnabled, imageWidth, imageHeight
static constexpr BindingSlot kAccumConstants = PassRootConstants("AccumCB", ACCUM_REG_CB, 4);
static constexpr BindingSlot kAccumVarianceM2 = PassUav("gVarianceM2", ACCUM_REG_VARIANCE_M2);

static constexpr BindingSlot kVarianceM2      = PassUav("gVarianceM2", VARIANCE_REDUCE_REG_M2);
static constexpr BindingSlot kVarianceResult  = PassUav("gResult", VARIANCE_REDUCE_REG_RESULT);
static constexpr BindingSlot kVarianceConstants = PassRootConstants("VarianceReduceCB", VARIANCE_REDUCE_REG_CB, 2);

void FrameAccumulationPass::CreateRootSignature()
{
    spdlog::debug("Creating root signature for accumulation pass");

    m_rootSignature = RootSignatureBuilder(L"FrameAccumulationPass Root Signature", /*tableCount*/ 2)
                          .Add(kAccumInputSlots)
                          .Add(kAccumTargetSlots)
                          .Add(kAccumConstants)
                          .Add(kAccumVarianceM2)
                          .Build(m_device.Get());

    m_varianceRootSignature = RootSignatureBuilder(L"VarianceReduce Root Signature", /*tableCount*/ 0)
                                  .Add(kVarianceM2)
                                  .Add(kVarianceResult)
                                  .Add(kVarianceConstants)
                                  .Build(m_device.Get());
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

    m_varianceM2 = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT4>>(
        m_device, static_cast<size_t>(width) * height, L"FrameAccumulation VarianceM2");
    m_varianceResult = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT2>>(
        m_device, 1, L"FrameAccumulation VarianceResult");

    if (!m_varianceReadback)
    {
        auto heapProps  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(DirectX::XMFLOAT2));
        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_varianceReadback)));
        m_varianceReadback->SetName(L"FrameAccumulation VarianceReadback");
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

    // This frame's own copy of the block. The ring is what makes writing a
    // shader-visible descriptor at record time safe: frame pacing has already
    // waited on the fence for the slot being reused, so no frame in flight can
    // still be reading these three.
    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    const uint32_t ringSlot = m_ringSlot;
    m_ringSlot = (m_ringSlot + 1) % Constants::Graphics::NUM_FRAMES;

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(currentFrameOutput.GetUnderlyingResource().Get(), &srvDesc,
        globalHeap.CpuHandle(GlobalDescriptor::AccumulationInput, ringSlot));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_accumulationBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc,
        globalHeap.CpuHandle(GlobalDescriptor::AccumulationTargets, ringSlot * 2));

    uavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    m_device->CreateUnorderedAccessView(m_displayBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc,
        globalHeap.CpuHandle(GlobalDescriptor::AccumulationTargets, ringSlot * 2 + 1));

    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_program->GetPipelineState());

    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_rootSignature.SetTable(m_commandList.Get(), 0, globalHeap.GpuHandle(GlobalDescriptor::AccumulationInput, ringSlot));
    m_rootSignature.SetTable(m_commandList.Get(), 1, globalHeap.GpuHandle(GlobalDescriptor::AccumulationTargets, ringSlot * 2));

    UINT width = Window::Get().GetWidth();
    UINT height = Window::Get().GetHeight();

    const uint32_t constants[4] = { m_frameCount + 1, m_varianceEnabled ? 1u : 0u, width, height };
    m_rootSignature.SetConstants(m_commandList.Get(), kAccumConstants, constants, 4);
    m_rootSignature.Set(m_commandList.Get(), kAccumVarianceM2, m_varianceM2->GetGPUVirtualAddress());

    UINT threadsX = (width + 7) / 8;   // 8x8 thread groups
    UINT threadsY = (height + 7) / 8;
    CommandContext::Get().Dispatch(threadsX, threadsY, 1);

    m_frameCount++;
}

void FrameAccumulationPass::RecordVarianceReduction()
{
    if (!m_initialized || !m_varianceEnabled || m_frameCount < 2)
        return;

    if (!m_varianceProgram)
        m_varianceProgram = ShaderProgramCache::Get().GetOrCreateCompute(m_device.Get(), m_varianceRootSignature.Get(),
            "resources/shaders/varianceReduce.cs.shader", L"VarianceReduce PSO");

    m_commandList->SetComputeRootSignature(m_varianceRootSignature.Get());
    m_commandList->SetPipelineState(m_varianceProgram->GetPipelineState());
    m_varianceRootSignature.Set(m_commandList.Get(), kVarianceM2, m_varianceM2->GetGPUVirtualAddress());
    m_varianceRootSignature.Set(m_commandList.Get(), kVarianceResult, m_varianceResult->GetGPUVirtualAddress());

    const uint32_t constants[2] = { static_cast<uint32_t>(m_varianceM2->GetElementsCount()), m_frameCount };
    m_varianceRootSignature.SetConstants(m_commandList.Get(), kVarianceConstants, constants, 2);

    CommandContext::Get().Dispatch(1, 1, 1);
}

// Separate node from the reduction so the graph, not this pass, places the
// UNORDERED_ACCESS -> COPY_SOURCE transition between them (ADR 0017).
void FrameAccumulationPass::RecordVarianceCopy()
{
    if (!m_initialized || !m_varianceEnabled || m_frameCount < 2)
        return;

    m_commandList->CopyResource(m_varianceReadback.Get(), m_varianceResult->GetUnderlyingResource().Get());
    m_varianceRecorded = true;
}

VarianceReadout FrameAccumulationPass::ReadVarianceResult()
{
    VarianceReadout readout;
    if (!m_varianceRecorded)
        return readout;
    m_varianceRecorded = false;

    void* mapped = nullptr;
    D3D12_RANGE readRange = { 0, sizeof(DirectX::XMFLOAT2) };
    ThrowIfFailed(m_varianceReadback->Map(0, &readRange, &mapped));
    const auto* values = static_cast<const DirectX::XMFLOAT2*>(mapped);
    readout.mean     = values->x;
    readout.relative = values->y;
    readout.valid    = true;
    D3D12_RANGE writeRange = { 0, 0 };
    m_varianceReadback->Unmap(0, &writeRange);
    return readout;
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
