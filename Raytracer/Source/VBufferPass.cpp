#include "pch.h"
#include "CommandContext.h"
#include "VBufferPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "FrameBindingLayout.h"
#include "GlobalDescriptorHeap.h"
#include "PassRegisters.h"
#include "Renderer.h"
#include "RootSignatureLibrary.h"
#include "Window.h"
#include "Resources/ShaderBindingTable.h"
#include "Resources/StructuredBuffer.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"

TechniqueDesc VBufferPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
        {"resources/shaders/vbufferPass.rg.shader", L"VBufferRayGen", ShaderRole::RayGen},
        {"resources/shaders/vbufferPass.ms.shader", L"VBufferMiss",   ShaderRole::Miss},
        {"resources/shaders/vbufferPass.ch.shader", L"VBufferHit",    ShaderRole::ClosestHit},
        {"resources/shaders/vbufferPass.ah.shader", L"VBufferAnyHit", ShaderRole::AnyHit},
    };
    desc.hitGroups = {
        {L"VBufferHitGroup", L"VBufferHit", L"VBufferAnyHit"},
    };
    desc.maxPayloadSize    = 4 * sizeof(float); // VBufferPayload: 2x uint + float2
    desc.maxAttributeSize  = 2 * sizeof(float);
    desc.maxRecursionDepth = 1;
    return desc;
}

// The frame layout plus one output: the packed primary-hit identity.
static constexpr BindingSlot kVBufferOutput = Accesses(
    PassTableEntry("gVBuffer", BindingKind::Uav, VBUFFER_REG_VBUFFER, GlobalDescriptor::VBuffer),
    GraphAccess::ComputeWrite);

void VBufferPass::DeclareGraphResources(RenderGraphPassBuilder& pass, const VxpgGraphHandles& vxpg) const
{
    pass.Declare(kVBufferOutput, vxpg.vbuffer);
}

void VBufferPass::CreateGlobalRootSignature()
{
    m_globalRootSignature = RootSignatureBuilder(L"VBuffer GlobalRootSig", /*tableCount*/ 1)
                                .AddFrameLayout()
                                .Add(kVBufferOutput)
                                .WithStaticSamplers()
                                .Build(m_device.Get());
}

void VBufferPass::CreateShaderResourceHeap()
{
    DxrPass::CreateShaderResourceHeap();

    // VBuffer texture (re)created here so window resize (which re-runs
    // CreateShaderResourceHeap) resizes it to the new render dimensions.
    CreateVBufferResource();
}

void VBufferPass::CreateVBufferResource()
{
    m_vbufferTex.Reset();

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Format           = DXGI_FORMAT_R32G32B32A32_UINT;
    desc.Width            = Window::Get().GetWidth();
    desc.Height           = Window::Get().GetHeight();
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_vbufferTex)));
    m_vbufferTex->SetName(L"VXPG VBuffer");

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format        = DXGI_FORMAT_R32G32B32A32_UINT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_vbufferTex.Get(), nullptr, &uavDesc,
        GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::VBuffer));
}

void VBufferPass::Render()
{
    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());

    FrameBindingLayout::Bind(m_commandList.Get(), m_globalRootSignature, *m_currentScene, *m_passConstants);

    D3D12_DISPATCH_RAYS_DESC desc = {};
    desc.RayGenerationShaderRecord.StartAddress = m_shaderBindingTable->GetUnderlyingResource()->GetGPUVirtualAddress();
    desc.RayGenerationShaderRecord.SizeInBytes  = m_shaderBindingTable->GetRayGenSectionSize();

    desc.MissShaderTable.StartAddress  = desc.RayGenerationShaderRecord.StartAddress + desc.RayGenerationShaderRecord.SizeInBytes;
    desc.MissShaderTable.StrideInBytes = m_shaderBindingTable->GetMissEntrySize();
    desc.MissShaderTable.SizeInBytes   = m_shaderBindingTable->GetMissSectionSize();

    desc.HitGroupTable.StartAddress  = desc.MissShaderTable.StartAddress + desc.MissShaderTable.SizeInBytes;
    desc.HitGroupTable.StrideInBytes = m_shaderBindingTable->GetHitEntrySize();
    desc.HitGroupTable.SizeInBytes   = m_shaderBindingTable->GetHitSectionSize();

    desc.Width  = Window::Get().GetWidth();
    desc.Height = Window::Get().GetHeight();
    desc.Depth  = 1;

    m_commandList->SetPipelineState1(m_rtStateObject.Get());
    CommandContext::Get().DispatchRays(desc);

    // No tail barrier: the graph emits it from the consumer's declaration.
}
