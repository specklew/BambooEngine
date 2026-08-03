#include "pch.h"
#include "RasterizationTechnique.h"

#include "CommandContext.h"
#include "GlobalDescriptorHeap.h"
#include "InputElements.h"
#include "PassRegisters.h"
#include "RasterDebugMode.h"
#include "ResourceManager/ResourceManager.h"
#include "Resources/ConstantBuffer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/GameObject.h"
#include "SceneResources/Material.h"
#include "SceneResources/Model.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/Scene.h"
#include "Shader.h"
#include "ShaderReflection.h"
#include "Utils/CVars.h"
#include "Utils/PassConstants.h"
#include "VoxelizationPass.h"

static AutoCVarEnum g_rasterizationDebugMode("renderer.rasterDebugMode", "Rasterization shader debug visualization mode", RasterDebugMode::None,
                                             CVarFlags::None, FormatDebugViewDocs<RasterDebugMode>(kRasterDebugModeDocs));

namespace
{
// Rasterization keeps its own layout rather than adopting the frame one: the
// raytracing output UAV, TLAS and merged vertex/index SRVs are deliberately
// absent, because reflection shows no rasterization shader touches them and
// carrying them would only cost root DWORDs. So every slot here is pass-scoped
// (space1) — except PassConstants, which really is a frame binding and stays at
// its space0 register, shared with passConstants.hlsl.
constexpr BindingSlot kCamera =
    PassTableEntry("CameraParams", BindingKind::Cbv, RASTER_REG_CAMERA_CB, GlobalDescriptor::CameraMatrices);
constexpr BindingSlot kTextures = PassTableEntry("gTextures", BindingKind::Srv, RASTER_REG_TEXTURES,
                                                 GlobalDescriptor::MaterialTextures, FRAME_MAX_TEXTURES);
constexpr BindingSlot kModelConstants     = PassCbv("ModelTransforms", RASTER_REG_MODEL_CB);
constexpr BindingSlot kMaterialConstants  = PassCbv("Material", RASTER_REG_MATERIAL_CB);
constexpr BindingSlot kPassConstants     = RootCbv("PassConstants", FRAME_REG_PASS_CONSTANTS);

constexpr BindingSlot kRasterSlots[] = {
    kCamera, kTextures, kModelConstants, kMaterialConstants, kPassConstants};
} // namespace

void RasterizationTechnique::SetFrameTargetFormats(DXGI_FORMAT backBufferFormat, DXGI_FORMAT depthStencilFormat)
{
    m_backBufferFormat   = backBufferFormat;
    m_depthStencilFormat = depthStencilFormat;
}

void RasterizationTechnique::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                                        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                                        std::shared_ptr<Scene> initialScene,
                                        std::shared_ptr<PassConstants> passConstants)
{
    m_device        = device;
    m_commandList   = commandList;
    m_scene         = std::move(initialScene);
    m_passConstants = std::move(passConstants);

    CreateRootSignature();
    CreatePipelineState();
}

void RasterizationTechnique::CreateRootSignature()
{
    m_rootSignature = RootSignatureBuilder(L"Rasterization RootSig", /*tableCount*/ 1)
                          .ForGraphics()
                          .WithFlags(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)
                          .Add(kRasterSlots)
                          .WithStaticSamplers()
                          .Build(m_device.Get());
}

void RasterizationTechnique::CreatePipelineState()
{
    auto& rm = ResourceManager::Get();

    auto psh = rm.GetOrLoadShader(AssetId("resources/shaders/colorShader.ps.shader"));
    m_pixelShader = rm.shaders.GetResource(psh).bytecode;
    auto vsh = rm.GetOrLoadShader(AssetId("resources/shaders/colorShader.vs.shader"));
    m_vertexShader = rm.shaders.GetResource(vsh).bytecode;

    ShaderReflection::ValidateShaderAsset("resources/shaders/colorShader.vs.shader", m_rootSignature.Get());
    ShaderReflection::ValidateShaderAsset("resources/shaders/colorShader.ps.shader", m_rootSignature.Get());

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
    desc.VS = {static_cast<BYTE*>(m_vertexShader->GetBufferPointer()), m_vertexShader->GetBufferSize()};
    desc.PS = {static_cast<BYTE*>(m_pixelShader->GetBufferPointer()), m_pixelShader->GetBufferSize()};
    desc.InputLayout = {inputLayout, _countof(inputLayout)};
    desc.pRootSignature = m_rootSignature.Get();

    CD3DX12_RASTERIZER_DESC rasterDesc(D3D12_DEFAULT);
    rasterDesc.FrontCounterClockwise = TRUE; // Loaders store canonical CCW winding; CCW is front-facing.
    desc.RasterizerState = rasterDesc;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);

    desc.SampleMask = UINT_MAX;

    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = m_backBufferFormat;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.DSVFormat = m_depthStencilFormat;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
    ThrowIfFailed(m_device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)));
    pso->SetName(L"Default Pipeline State");
    m_pipelineStateObject = pso;
}

// Surface views read the material and the interpolated vertex data, nothing the
// VXPG chain produces — every buffer view moved to DebugViewPass.
bool RasterizationTechnique::UsesVoxelGuiding() const
{
    return false;
}

int RasterizationTechnique::GetDebugMode() const
{
    return static_cast<int>(g_rasterizationDebugMode.Get());
}

std::vector<RenderTechnique::DebugView> RasterizationTechnique::GetDebugViews() const
{
    return WithBufferViews(BuildDebugViews<RasterDebugMode>(kRasterDebugModeDocs));
}

bool RasterizationTechnique::SetDebugView(int index)
{
    const auto mode = magic_enum::enum_cast<RasterDebugMode>(index);
    if (!mode)
        return false;
    g_rasterizationDebugMode.Set(*mode);
    return true;
}

GraphResourceHandle RasterizationTechnique::BuildGraph(RenderGraph& graph, const FrameGraphContext& frame)
{
    // Raytracing overwrites the whole back buffer with the present copy and draws
    // no geometry, so both clears only exist on this path (measured 2026-08-01,
    // ABeautifulGame 3 s Debug: PT 3151 -> 3276 frames without them).
    graph.AddPass("Raster Clear",
        [&](RenderGraphPassBuilder& pass)
        {
            pass.Write(frame.backBuffer, GraphAccess::RenderTarget);
            pass.Write(frame.depthStencil, GraphAccess::DepthWrite);
        },
        [this, frame]() { Clear(frame); });

    graph.AddPass("Raster Draw",
        [&](RenderGraphPassBuilder& pass)
        {
            pass.Write(frame.backBuffer, GraphAccess::RenderTarget);
            pass.Write(frame.depthStencil, GraphAccess::DepthWrite);
        },
        [this, frame]() { DrawScene(frame); });

    // The back buffer already holds the image; the frame needs no display chain.
    return InvalidGraphResource;
}

void RasterizationTechnique::Clear(const FrameGraphContext& frame) const
{
    constexpr FLOAT clearColor[4] = { 0.3f, 0.6f, 0.9f, 1.0f };
    ID3D12GraphicsCommandList4* commandList = CommandContext::Get().GetCommandList();

    commandList->ClearRenderTargetView(frame.backBufferRtv, clearColor, 0, nullptr);
    commandList->ClearDepthStencilView(frame.depthStencilDsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

void RasterizationTechnique::DrawScene(const FrameGraphContext& frame) const
{
    ID3D12GraphicsCommandList4* commandList = CommandContext::Get().GetCommandList();
    ID3D12DescriptorHeap* descriptorHeaps[] = { GlobalDescriptorHeap::Get().GetHeap() };

    commandList->OMSetRenderTargets(1, &frame.backBufferRtv, true, &frame.depthStencilDsv);
    commandList->SetPipelineState(m_pipelineStateObject.Get());
    commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
    commandList->SetGraphicsRootSignature(m_rootSignature.Get());

    m_rootSignature.Set(commandList, kPassConstants, m_passConstants->GetGpuVirtualAddress());

    for (const auto& go : m_scene->GetGameObjects())
    {
        auto gpuAddress = go->GetWorldMatrixBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress();
        m_rootSignature.Set(commandList, kModelConstants, gpuAddress);

        for (const auto& primitive : go->GetModel()->GetMeshes())
        {
            gpuAddress = primitive->m_material->m_materialBuffer->GetUnderlyingResource()->GetGPUVirtualAddress();
            m_rootSignature.Set(commandList, kMaterialConstants, gpuAddress);

            auto vertex_view = primitive->GetVertexView();
            auto index_view = primitive->GetIndexView();

            auto vertexBuffer = std::dynamic_pointer_cast<VertexBuffer>(vertex_view.buffer);
            auto indexBuffer = std::dynamic_pointer_cast<IndexBuffer>(index_view.buffer);

            commandList->IASetVertexBuffers(0, 1, &vertexBuffer->GetVertexBufferView());
            commandList->IASetIndexBuffer(&indexBuffer->GetIndexBufferView());
            commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            m_rootSignature.SetTable(commandList, 0, GlobalDescriptorHeap::Get().GpuStart());

            CommandContext::Get().DrawIndexedInstanced(index_view.count, 1, index_view.offset, vertex_view.offset, 0);
        }
    }
}

REGISTER_TECHNIQUE("Rasterization", RasterizationTechnique)
