#include "pch.h"
#include "CommandContext.h"
#include "PostProcessPass.h"

#include "Constants.h"
#include "PassRegisters.h"
#include "RootSignatureLibrary.h"
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

    CreateResources();
    CreateRootSignature();
    CreatePSO();

    m_initialized = true;
    spdlog::info("Post-process pass initialized successfully.");
}

// Accumulated image in, tone-mapped image out. Two tables for the same reason as
// the accumulation pass: SRV and UAV are separate NUM_FRAMES-deep blocks of the
// global heap, and the offsets here are relative to the frame slot's base.
static constexpr BindingSlot kPostInputSlots[] = {
    PassTableEntryAt("gInput", BindingKind::Srv, POST_REG_INPUT, 0, 1, /*tableIndex*/ 0),
};
static constexpr BindingSlot kPostOutputSlots[] = {
    PassTableEntryAt("gOutput", BindingKind::Uav, POST_REG_OUTPUT, 0, 1, /*tableIndex*/ 1),
};
static constexpr BindingSlot kPostConstants = PassRootConstants("PostProcessCB", POST_REG_CB, 4);

void PostProcessPass::CreateRootSignature()
{
    spdlog::debug("Creating root signature for post-process pass");

    m_rootSignature = RootSignatureBuilder(L"PostProcessPass Root Signature", /*tableCount*/ 2)
                          .Add(kPostInputSlots)
                          .Add(kPostOutputSlots)
                          .Add(kPostConstants)
                          .Build(m_device.Get());
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

        m_outputBuffer = std::make_unique<Texture>(m_device, desc,
            D3D12_RESOURCE_STATE_COPY_SOURCE, L"PostProcess Output Buffer");
    }
}

void PostProcessPass::CreatePSO()
{
    spdlog::debug("Creating PSO for post-process compute shader");

    m_program = ShaderProgramCache::Get().GetOrCreateCompute(m_device.Get(), m_rootSignature.Get(),
        "resources/shaders/postprocess.cs.shader", L"PostProcessPass PSO");
}

void PostProcessPass::Dispatch(Texture& input, const PostProcessParams& params)
{
    if (!m_initialized)
        return;

    // This frame's copy of the block: the input is whichever image the frame
    // produced, so it cannot be written once at creation, and only the ring makes
    // a record-time write safe (see FrameAccumulationPass).
    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    const uint32_t ringSlot = m_ringSlot;
    m_ringSlot = (m_ringSlot + 1) % Constants::Graphics::NUM_FRAMES;

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(input.GetUnderlyingResource().Get(), &srvDesc,
        globalHeap.CpuHandle(GlobalDescriptor::PostProcessInput, ringSlot));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc,
        globalHeap.CpuHandle(GlobalDescriptor::PostProcessOutput, ringSlot));

    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_program->GetPipelineState());

    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_rootSignature.SetTable(m_commandList.Get(), 0, globalHeap.GpuHandle(GlobalDescriptor::PostProcessInput, ringSlot));
    m_rootSignature.SetTable(m_commandList.Get(), 1, globalHeap.GpuHandle(GlobalDescriptor::PostProcessOutput, ringSlot));

    static_assert(sizeof(PostProcessParams) == 4 * sizeof(uint32_t), "PostProcessParams must be exactly 4 floats");
    m_rootSignature.SetConstants(m_commandList.Get(), kPostConstants, &params, 4);

    UINT width = Window::Get().GetWidth();
    UINT height = Window::Get().GetHeight();
    UINT threadsX = (width + 7) / 8;   // 8x8 thread groups
    UINT threadsY = (height + 7) / 8;
    CommandContext::Get().Dispatch(threadsX, threadsY, 1);
}

// Separate graph node from the dispatch: the output flips UAV -> COPY_SOURCE and
// the back buffer RENDER_TARGET -> COPY_DEST between the two, which the graph
// synthesizes from the declarations.
void PostProcessPass::CopyToBackBuffer(Texture& backBuffer)
{
    if (!m_initialized)
        return;

    CommandContext::Get().CopyResource(backBuffer.GetUnderlyingResource().Get(),
        m_outputBuffer->GetUnderlyingResource().Get());
}

void PostProcessPass::OnResize()
{
    if (!m_initialized)
        return;

    spdlog::debug("Resizing post-process output buffer");
    m_outputBuffer.reset();
    CreateResources();
}
