#include "pch.h"
#include "CommandContext.h"
#include "LightInjectionPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "FrameBindingLayout.h"
#include "GlobalDescriptorHeap.h"
#include "PassRegisters.h"
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

// The frame layout plus the injection outputs: the voxel accumulators it writes,
// the G-buffers it fills for the guiding passes, and the grid constants.
namespace
{
constexpr BindingSlot kVoxelIrradiance = GraphWrites(
    PassTableEntry("gVoxelIrradiance", BindingKind::Uav, INJECT_REG_IRRADIANCE, GlobalDescriptor::VoxelIrradiance),
    GraphAccess::ComputeWrite);
constexpr BindingSlot kVoxelVplCount = GraphWrites(
    PassTableEntry("gVoxelVplCount", BindingKind::Uav, INJECT_REG_VPL_COUNT, GlobalDescriptor::VoxelVplCount),
    GraphAccess::ComputeWrite);
constexpr BindingSlot kShadingPoints = GraphWrites(
    PassTableEntry("gShadingPoints", BindingKind::Uav, INJECT_REG_SHADING_POINTS, GlobalDescriptor::ShadingPoints),
    GraphAccess::ComputeWrite);
constexpr BindingSlot kVoxelRepresentative = GraphWrites(
    PassTableEntry("gVoxelRepresentative", BindingKind::Uav, INJECT_REG_VOXEL_REPRESENTATIVE,
                   GlobalDescriptor::VoxelRepresentative),
    GraphAccess::ComputeWrite);
constexpr BindingSlot kVplPosition = GraphWrites(
    PassTableEntry("gVplPosition", BindingKind::Uav, INJECT_REG_VPL_POSITION, GlobalDescriptor::VplPosition),
    GraphAccess::ComputeWrite);
constexpr BindingSlot kVBuffer = GraphReads(
    PassTableEntry("gVBuffer", BindingKind::Uav, INJECT_REG_VBUFFER, GlobalDescriptor::VBuffer),
    GraphAccess::UnorderedAccessRead);
// The grid dimensions, not a graph resource: no producer inside the frame.
constexpr BindingSlot kVoxelGridConstants = PassCbv("VoxelGridCB", REG_VOXEL_GRID_CB);

constexpr BindingSlot kInjectionSlots[] = {
    kVoxelIrradiance, kVoxelVplCount, kShadingPoints, kVoxelRepresentative, kVplPosition, kVBuffer};
}

void LightInjectionPass::DeclareGraphResources(RenderGraphPassBuilder& pass, const VxpgGraphHandles& vxpg) const
{
    // Same order as kInjectionSlots; the slot supplies the access and direction,
    // this only says which of the frame's resources sits behind each binding.
    const GraphResourceHandle handles[] = {
        vxpg.voxelIrradiance, vxpg.voxelVplCount, vxpg.shadingPoints,
        vxpg.voxelRepresentative, vxpg.vplPosition, vxpg.vbuffer};
    static_assert(std::size(handles) == std::size(kInjectionSlots), "one handle per graph-visible slot");

    for (size_t i = 0; i < std::size(kInjectionSlots); ++i)
        pass.Declare(kInjectionSlots[i], handles[i]);
}

void LightInjectionPass::CreateGlobalRootSignature()
{
    m_globalRootSignature = RootSignatureBuilder(L"LightInjection GlobalRootSig", /*tableCount*/ 1)
                                .AddFrameLayout()
                                .Add(kVoxelIrradiance)
                                .Add(kVoxelVplCount)
                                .Add(kShadingPoints)
                                .Add(kVoxelRepresentative)
                                .Add(kVplPosition)
                                .Add(kVBuffer)
                                .Add(kVoxelGridConstants)
                                .WithStaticSamplers()
                                .Build(m_device.Get());
}

void LightInjectionPass::CreateShaderResourceHeap()
{
    DxrPass::CreateShaderResourceHeap();

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

    FrameBindingLayout::Bind(m_commandList.Get(), m_globalRootSignature, *m_currentScene, *m_passConstants);
    m_globalRootSignature.Set(m_commandList.Get(), kVoxelGridConstants,
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
