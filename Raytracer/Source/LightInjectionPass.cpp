#include "pch.h"
#include "CommandContext.h"
#include "LightInjectionPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "GlobalDescriptorHeap.h"
#include "Renderer.h"
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
    // Mirrors RaytracePass::CreateGlobalRootSignature, extended with the voxel
    // irradiance/count UAVs (u1/u2) and the voxel grid constants CBV (b4).

    D3D12_DESCRIPTOR_RANGE cbvRange;
    cbvRange.BaseShaderRegister = 0;
    cbvRange.NumDescriptors = 1;
    cbvRange.RegisterSpace = 0;
    cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    cbvRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::CameraMatrices);

    D3D12_DESCRIPTOR_RANGE rtRange;
    rtRange.BaseShaderRegister = 0;
    rtRange.NumDescriptors = 1;
    rtRange.RegisterSpace = 0;
    rtRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    rtRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::RaytraceOutput);

    D3D12_DESCRIPTOR_RANGE tlasRange;
    tlasRange.BaseShaderRegister = 0;
    tlasRange.NumDescriptors = 1;
    tlasRange.RegisterSpace = 0;
    tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    tlasRange.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Tlas);

    D3D12_DESCRIPTOR_RANGE vertex_range;
    vertex_range.BaseShaderRegister = 1;
    vertex_range.NumDescriptors = 1;
    vertex_range.RegisterSpace = 0;
    vertex_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    vertex_range.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Vertices);

    D3D12_DESCRIPTOR_RANGE index_range;
    index_range.BaseShaderRegister = 2;
    index_range.NumDescriptors = 1;
    index_range.RegisterSpace = 0;
    index_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    index_range.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Indices);

    D3D12_DESCRIPTOR_RANGE texture_range;
    texture_range.BaseShaderRegister = 7;
    texture_range.NumDescriptors = Constants::Graphics::MAX_TEXTURES;
    texture_range.RegisterSpace = 0;
    texture_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    texture_range.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::MaterialTextures);

    D3D12_DESCRIPTOR_RANGE skybox_range;
    skybox_range.BaseShaderRegister = 0;
    skybox_range.NumDescriptors = 1;
    skybox_range.RegisterSpace = 1;
    skybox_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    skybox_range.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(GlobalDescriptor::Skybox);

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

    D3D12_DESCRIPTOR_RANGE ranges[13] = {cbvRange, rtRange, tlasRange, vertex_range, index_range,
                                        texture_range, skybox_range, voxelIrradianceRange, voxelVplCountRange,
                                        shadingPointsRange, voxelRepresentativeRange, vplPositionRange,
                                        vbufferRange};

    CD3DX12_ROOT_PARAMETER rootParameters[10];
    rootParameters[0].InitAsDescriptorTable(13, ranges);
    rootParameters[1].InitAsShaderResourceView(3, 0); // Geometry Info
    rootParameters[2].InitAsShaderResourceView(4, 0); // Instance Info
    rootParameters[3].InitAsShaderResourceView(5, 0); // Random buffer
    rootParameters[4].InitAsShaderResourceView(6, 0); // Lights struct buffer
    rootParameters[5].InitAsConstants(1, 1);           // Time
    rootParameters[6].InitAsConstantBufferView(3, 0);  // Pass constants
    rootParameters[7].InitAsConstantBufferView(4, 0);  // Voxel grid constants
    rootParameters[8].InitAsShaderResourceView(1, 1); // Emissive triangles (t1, space1)
    rootParameters[9].InitAsShaderResourceView(2, 1); // Light pool (t2, space1)

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(10, rootParameters);

    auto static_samplers = Renderer::GetStaticSamplers();
    rootSignatureDesc.NumStaticSamplers = static_cast<UINT>(static_samplers.size());
    rootSignatureDesc.pStaticSamplers   = static_samplers.data();
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSignature)));
    m_globalRootSignature->SetName(L"LightInjection GlobalRootSig");
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

    ID3D12DescriptorHeap* heaps[] = {GlobalDescriptorHeap::Get().GetHeap()};
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootDescriptorTable(0, GlobalDescriptorHeap::Get().GpuStart());
    m_commandList->SetComputeRootShaderResourceView(1, m_currentScene->GetGeometryInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(2, m_currentScene->GetInstanceInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(3, m_randomBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(4, m_currentScene->GetLightDataBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());

    uint32_t time;
    memcpy(&time, &m_time, sizeof(float));
    m_commandList->SetComputeRoot32BitConstant(5, time, 0);
    m_commandList->SetComputeRootConstantBufferView(6, m_passConstants->GetGpuVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(7, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(8, m_currentScene->GetEmissiveTriangleBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(9, m_currentScene->GetLightPoolBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());

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

    CommandContext& context = CommandContext::Get();
    context.UavBarrierRaw(m_voxelPass->GetIrradianceTexture().Get());
    context.UavBarrierRaw(m_voxelPass->GetVplCountTexture().Get());
    context.UavBarrierRaw(m_shadingPointsTex.Get());
    context.UavBarrierRaw(m_voxelRepresentativeTex.Get());
    context.UavBarrierRaw(m_vplPositionTex.Get());
}
