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

// Private heap: accumulated image in, tone-mapped image out.
static constexpr BindingSlot kPostSlots[] = {
    PassTableEntryAt("gInput", BindingKind::Srv, POST_REG_INPUT, 0),
    PassTableEntryAt("gOutput", BindingKind::Uav, POST_REG_OUTPUT, 1),
};
static constexpr BindingSlot kPostConstants = PassRootConstants("PostProcessCB", POST_REG_CB, 4);

void PostProcessPass::CreateRootSignature()
{
    spdlog::debug("Creating root signature for post-process pass");

    m_rootSignature = RootSignatureBuilder(L"PostProcessPass Root Signature", /*tableCount*/ 1)
                          .Add(kPostSlots)
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

    // Get descriptor handles
    UINT descriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetCPUDescriptorHandleForHeapStart());

    // Create SRV for input at slot 0 (t0)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(input.GetUnderlyingResource().Get(), &srvDesc, cpuHandle);

    // Create UAV for output buffer at slot 1 (u0)
    cpuHandle.Offset(1, descriptorSize);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc, cpuHandle);

    // Bind root signature and PSO
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_program->GetPipelineState());

    // Set descriptor heap and bind descriptor table
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_rootSignature.SetTable(m_commandList.Get(), 0, m_descriptorHeap->GetGPUDescriptorHandleForHeapStart());

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
