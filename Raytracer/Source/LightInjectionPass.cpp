#include "pch.h"
#include "CommandContext.h"
#include "LightInjectionPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "FrameBindingLayout.h"
#include "GlobalDescriptorHeap.h"
#include "Renderer.h"
#include "RootSignatureLibrary.h"
#include "VoxelizationPass.h"
#include "Window.h"
#include "Resources/ShaderBindingTable.h"
#include "Resources/StructuredBuffer.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"

TechniqueDesc LightInjectionPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
        {"resources/shaders/lightInjection.rg.shader",      L"InjectRayGen", ShaderRole::RayGen},
        {"resources/shaders/lightInjection.ms.shader",      L"InjectMiss",   ShaderRole::Miss},
        {"resources/shaders/raytracing.shadowmiss.shader",  L"ShadowMiss",   ShaderRole::Miss},
        {"resources/shaders/lightInjection.ch.shader",      L"InjectHit",    ShaderRole::ClosestHit},
        {"resources/shaders/lightInjection.ah.shader",      L"InjectAnyHit", ShaderRole::AnyHit},
        {"resources/shaders/raytracing.shadowhit.shader",   L"ShadowHit",    ShaderRole::AnyHit},
    };
    // Order matters: shadow shaders must land at miss index 1 / hit group index 1
    // because TraceShadow() hardcodes those SBT offsets.
    desc.hitGroups = {
        {L"InjectHitGroup", L"InjectHit", L"InjectAnyHit"},
        {L"ShadowHitGroup", L"",          L"ShadowHit"},
    };
    desc.maxPayloadSize    = 9 * sizeof(float); // InjectPayload: 2x float3 + 3x uint
    desc.maxAttributeSize  = 2 * sizeof(float);
    desc.maxRecursionDepth = 3; // -> hit -> shadow ray
    return desc;
}

void LightInjectionPass::CreateGlobalRootSignature()
{
    // The frame layout, extended with the voxel irradiance/count UAVs (u1/u2), the
    // injection G-buffers, and the voxel grid constants CBV (b4).
    constexpr uint32_t VoxelGridConstantsCbv = FrameBindingLayout::kPassRootParameterStart;

    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    FrameBindingLayout::AppendFrameRanges(ranges);

    D3D12_DESCRIPTOR_RANGE voxelIrradianceRange;
    voxelIrradianceRange.BaseShaderRegister = 1;
    voxelIrradianceRange.NumDescriptors = 1;
    voxelIrradianceRange.RegisterSpace = 0;
    voxelIrradianceRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelIrradianceRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelIrradiance);

    D3D12_DESCRIPTOR_RANGE voxelVplCountRange;
    voxelVplCountRange.BaseShaderRegister = 2;
    voxelVplCountRange.NumDescriptors = 1;
    voxelVplCountRange.RegisterSpace = 0;
    voxelVplCountRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelVplCountRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelVplCount);

    D3D12_DESCRIPTOR_RANGE shadingPointsRange;
    shadingPointsRange.BaseShaderRegister = 3; // u3
    shadingPointsRange.NumDescriptors = 1;
    shadingPointsRange.RegisterSpace = 0;
    shadingPointsRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    shadingPointsRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::ShadingPoints);

    D3D12_DESCRIPTOR_RANGE voxelRepresentativeRange;
    voxelRepresentativeRange.BaseShaderRegister = 4; // u4
    voxelRepresentativeRange.NumDescriptors = 1;
    voxelRepresentativeRange.RegisterSpace = 0;
    voxelRepresentativeRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelRepresentativeRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VoxelRepresentative);

    D3D12_DESCRIPTOR_RANGE vplPositionRange;
    vplPositionRange.BaseShaderRegister = 5; // u5
    vplPositionRange.NumDescriptors = 1;
    vplPositionRange.RegisterSpace = 0;
    vplPositionRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    vplPositionRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VplPosition);

    D3D12_DESCRIPTOR_RANGE vbufferRange;
    vbufferRange.BaseShaderRegister = 6; // u6
    vbufferRange.NumDescriptors = 1;
    vbufferRange.RegisterSpace = 0;
    vbufferRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    vbufferRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::VBuffer);

    for (const D3D12_DESCRIPTOR_RANGE& range : {voxelIrradianceRange, voxelVplCountRange, shadingPointsRange,
                                                voxelRepresentativeRange, vplPositionRange, vbufferRange})
        ranges.push_back(range);

    CD3DX12_ROOT_PARAMETER rootParameters[FrameBindingLayout::kPassRootParameterStart + 1];
    FrameBindingLayout::FillFrameRootParameters(rootParameters, ranges);
    rootParameters[VoxelGridConstantsCbv].InitAsConstantBufferView(4, 0);

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(_countof(rootParameters), rootParameters);

    auto static_samplers = Renderer::GetStaticSamplers();
    rootSignatureDesc.NumStaticSamplers = static_cast<UINT>(static_samplers.size());
    rootSignatureDesc.pStaticSamplers   = static_samplers.data();
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    m_globalRootSignature = RootSignatureLibrary::Get().Create(m_device.Get(), rootSignatureDesc,
                                                               L"LightInjection GlobalRootSig", true);
}

void LightInjectionPass::CreateShaderResourceHeap()
{
    // TLAS only — the main raytrace pass owns the output UAV slot.
    const D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::Tlas);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format                                   = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = m_currentScene->GetAccelerationStructures()->GetTopLevelAS().p_result->GetGPUVirtualAddress();
    m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

    // ShadingPoints G-buffer (re)created here so window resize (which re-runs
    // CreateShaderResourceHeap) resizes it to the new render dimensions.
    CreateShadingPointsResource();
    CreateRepresentativeResources();
}

void LightInjectionPass::CreateShadingPointsResource()
{
    m_shadingPointsTex.Reset();

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Format           = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.Width            = Window::Get().GetWidth();
    desc.Height           = Window::Get().GetHeight();
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    // Created in UNORDERED_ACCESS (matches VoxelizationPass textures): written as
    // a UAV by injection and read as a UAV by the raster debug overlay, so it
    // stays in this layout — UAV barriers between writer/reader handle ordering.
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_shadingPointsTex)));
    m_shadingPointsTex->SetName(L"VXPG ShadingPoints");

    // UAV at the shared heap's ShadingPoints slot — bound via the global root
    // signature's u3 range so the injection raygen/closest-hit can write it.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format        = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_shadingPointsTex.Get(), nullptr, &uavDesc,
        GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::ShadingPoints));
}

void LightInjectionPass::CreateRepresentativeResources()
{
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();

    // Per-voxel representative VPL (pos + octa normal): grid-sized Texture3D.
    m_voxelRepresentativeTex.Reset();
    const uint32_t gridDim = m_voxelPass ? m_voxelPass->GetGridDim() : 64u;
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        desc.Format           = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.Width            = gridDim;
        desc.Height           = gridDim;
        desc.DepthOrArraySize = static_cast<UINT16>(gridDim);
        desc.MipLevels        = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_voxelRepresentativeTex)));
        m_voxelRepresentativeTex->SetName(L"VXPG VoxelRepresentative");

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format               = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.ViewDimension        = D3D12_UAV_DIMENSION_TEXTURE3D;
        uavDesc.Texture3D.WSize      = gridDim;
        m_device->CreateUnorderedAccessView(m_voxelRepresentativeTex.Get(), nullptr, &uavDesc,
            globalHeap.CpuHandle(GlobalDescriptor::VoxelRepresentative));
    }

    // Per-pixel VPL hit position: screen-sized Texture2D (mirrors ShadingPoints).
    m_vplPositionTex.Reset();
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Format           = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.Width            = Window::Get().GetWidth();
        desc.Height           = Window::Get().GetHeight();
        desc.DepthOrArraySize = 1;
        desc.MipLevels        = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_vplPositionTex)));
        m_vplPositionTex->SetName(L"VXPG VplPosition");

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format        = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(m_vplPositionTex.Get(), nullptr, &uavDesc,
            globalHeap.CpuHandle(GlobalDescriptor::VplPosition));
    }
}

void LightInjectionPass::Render()
{
    if (!m_voxelPass)
        return;

    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());

    FrameBindingLayout::BindFrameRootParameters(m_commandList.Get(), *m_currentScene, *m_passConstants);
    m_commandList->SetComputeRootConstantBufferView(FrameBindingLayout::kPassRootParameterStart,
                                                   m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());

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

    // No tail barriers: every output this pass writes (ShadingPoints, irradiance,
    // VPL count, representative, VPL position) is declared on the graph, which
    // emits the barrier where a reader actually needs it.
}
