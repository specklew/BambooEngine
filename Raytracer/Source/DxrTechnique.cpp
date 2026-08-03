#include "pch.h"
#include "DxrTechnique.h"

#include "GlobalDescriptorHeap.h"
#include "RaytraceDebugMode.h"
#include "Utils/CVars.h"
#include "Window.h"

static AutoCVarEnum g_raytraceDebugMode("renderer.raytraceDebugMode", "Raytracing shader debug visualization mode", RaytraceDebugMode::None,
                                        CVarFlags::None, FormatDebugViewDocs<RaytraceDebugMode>(kRaytraceDebugModeDocs));

int DxrTechnique::GetDebugMode() const
{
    return static_cast<int>(g_raytraceDebugMode.Get());
}

bool DxrTechnique::HasActiveDebugView() const
{
    return g_raytraceDebugMode.Get() != RaytraceDebugMode::None;
}

void DxrTechnique::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                              Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                              std::shared_ptr<Scene> initialScene,
                              std::shared_ptr<PassConstants> passConstants)
{
    // The base writes the output UAV into the shared heap as part of its own
    // setup, so the texture has to exist before it runs; m_device is set here
    // rather than waited for because CreateOutputTexture needs it.
    m_device = device;
    CreateOutputTexture();

    DxrPass::Initialize(device, commandList, initialScene, passConstants);
}

void DxrTechnique::OnResize()
{
    CreateOutputTexture();
    DxrPass::OnResize();
}

bool DxrTechnique::SetDebugViewsCompiled(bool enabled)
{
    if (m_compileDebugViews == enabled)
        return false;
    m_compileDebugViews = enabled;
    return true;
}

bool DxrTechnique::SetOneSampleMisCompiled(bool enabled)
{
    if (m_compileOneSampleMis == enabled)
        return false;
    m_compileOneSampleMis = enabled;
    return true;
}

GraphResourceHandle DxrTechnique::BuildGraph(RenderGraph& graph, const FrameGraphContext& frame)
{
    m_frameGuiding = frame.voxelGuiding;

    const GraphResourceHandle output = graph.Import(*m_outputResource, "Raytrace Output");

    graph.AddPass("Raytrace Technique",
        [&](RenderGraphPassBuilder& pass)
        {
            pass.Write(output, GraphAccess::ComputeWrite);
            DeclareDispatchResources(graph, pass);
        },
        [this]() { Render(); });

    AppendPostDispatchNodes(graph);

    return output;
}

void DxrTechnique::CreateOutputTexture()
{
    spdlog::debug("Creating raytracing output buffer");

    m_outputResource.reset();

    D3D12_RESOURCE_DESC outputBufferDesc = {};
    outputBufferDesc.DepthOrArraySize = 1;
    outputBufferDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    outputBufferDesc.Format           = DXGI_FORMAT_R16G16B16A16_FLOAT;
    outputBufferDesc.Width            = Window::Get().GetWidth();
    outputBufferDesc.Height           = Window::Get().GetHeight();
    outputBufferDesc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    outputBufferDesc.MipLevels        = 1;
    outputBufferDesc.SampleDesc.Count = 1;
    outputBufferDesc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    m_outputResource = std::make_unique<Texture>(m_device, outputBufferDesc,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"Raytrace Output");
}

void DxrTechnique::CreateShaderResourceHeap()
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputResource->GetUnderlyingResource().Get(), nullptr, &uavDesc,
        GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::RaytraceOutput));

    DxrPass::CreateShaderResourceHeap();
}
