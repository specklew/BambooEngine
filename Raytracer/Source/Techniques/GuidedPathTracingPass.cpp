#include "pch.h"
#include "Techniques/GuidedPathTracingPass.h"

#include "AccelerationStructures.h"
#include "Constants.h"
#include "Renderer.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "Window.h"
#include "Resources/ShaderBindingTable.h"
#include "Resources/StructuredBuffer.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"

TechniqueDesc GuidedPathTracingPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
            {"resources/shaders/guidedPathTracing.rg.shader",   L"GuidedRayGen", ShaderRole::RayGen},
            {"resources/shaders/guidedPathTracing.ms.shader",   L"GuidedMiss",   ShaderRole::Miss},
            {"resources/shaders/raytracing.shadowmiss.shader",  L"ShadowMiss",   ShaderRole::Miss},
            {"resources/shaders/guidedPathTracing.ch.shader",   L"GuidedHit",    ShaderRole::ClosestHit},
            {"resources/shaders/guidedPathTracing.ah.shader",   L"GuidedAnyHit", ShaderRole::AnyHit},
            {"resources/shaders/raytracing.shadowhit.shader",   L"ShadowHit",    ShaderRole::AnyHit},
    };
    // Shadow shaders must land at miss index 1 / hit group index 1 (TraceShadow
    // hardcodes those SBT offsets).
    desc.hitGroups = {
        {L"GuidedHitGroup", L"GuidedHit", L"GuidedAnyHit"},
        {L"ShadowHitGroup", L"",          L"ShadowHit"},
    };
    desc.maxPayloadSize    = 9 * sizeof(float); // GuidedPayload: 2x float3 + 3x uint
    desc.maxAttributeSize  = 2 * sizeof(float);
    // Vanilla chain depth + shadow rays at the terminal vertex; extra headroom
    // because the first vertex spawns two sequential MIS subtrees.
    desc.maxRecursionDepth = 10;
    return desc;
}

void GuidedPathTracingPass::CreateGlobalRootSignature()
{
    // Base 7-param scene binding extended with: voxel irradiance/count UAVs
    // (u1/u2, global heap slots 520/521), voxel grid CBV (b4), and root UAVs
    // for the guiding distribution buffers (u3 counters, u4 ids, u5 cdf).

    D3D12_DESCRIPTOR_RANGE cbvRange;
    cbvRange.BaseShaderRegister = 0;
    cbvRange.NumDescriptors = 1;
    cbvRange.RegisterSpace = 0;
    cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    cbvRange.OffsetInDescriptorsFromTableStart = 1;

    D3D12_DESCRIPTOR_RANGE rtRange;
    rtRange.BaseShaderRegister = 0;
    rtRange.NumDescriptors = 1;
    rtRange.RegisterSpace = 0;
    rtRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    rtRange.OffsetInDescriptorsFromTableStart = 2;

    D3D12_DESCRIPTOR_RANGE tlasRange;
    tlasRange.BaseShaderRegister = 0;
    tlasRange.NumDescriptors = 1;
    tlasRange.RegisterSpace = 0;
    tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    tlasRange.OffsetInDescriptorsFromTableStart = 3;

    D3D12_DESCRIPTOR_RANGE vertex_range;
    vertex_range.BaseShaderRegister = 1;
    vertex_range.NumDescriptors = 1;
    vertex_range.RegisterSpace = 0;
    vertex_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    vertex_range.OffsetInDescriptorsFromTableStart = 4;

    D3D12_DESCRIPTOR_RANGE index_range;
    index_range.BaseShaderRegister = 2;
    index_range.NumDescriptors = 1;
    index_range.RegisterSpace = 0;
    index_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    index_range.OffsetInDescriptorsFromTableStart = 5;

    D3D12_DESCRIPTOR_RANGE texture_range;
    texture_range.BaseShaderRegister = 7;
    texture_range.NumDescriptors = Constants::Graphics::MAX_TEXTURES;
    texture_range.RegisterSpace = 0;
    texture_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    texture_range.OffsetInDescriptorsFromTableStart = 6;

    D3D12_DESCRIPTOR_RANGE skybox_range;
    skybox_range.BaseShaderRegister = 0;
    skybox_range.NumDescriptors = 1;
    skybox_range.RegisterSpace = 1;
    skybox_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    skybox_range.OffsetInDescriptorsFromTableStart = Constants::Graphics::SKYBOX_DESCRIPTOR_INDEX;

    D3D12_DESCRIPTOR_RANGE voxelIrradianceRange;
    voxelIrradianceRange.BaseShaderRegister = 1;
    voxelIrradianceRange.NumDescriptors = 1;
    voxelIrradianceRange.RegisterSpace = 0;
    voxelIrradianceRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelIrradianceRange.OffsetInDescriptorsFromTableStart = Constants::Graphics::VOXEL_IRRADIANCE_DESCRIPTOR_INDEX;

    D3D12_DESCRIPTOR_RANGE voxelVplCountRange;
    voxelVplCountRange.BaseShaderRegister = 2;
    voxelVplCountRange.NumDescriptors = 1;
    voxelVplCountRange.RegisterSpace = 0;
    voxelVplCountRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    voxelVplCountRange.OffsetInDescriptorsFromTableStart = Constants::Graphics::VOXEL_VPL_COUNT_DESCRIPTOR_INDEX;

    D3D12_DESCRIPTOR_RANGE ranges[9] = {cbvRange, rtRange, tlasRange, vertex_range, index_range,
                                        texture_range, skybox_range, voxelIrradianceRange, voxelVplCountRange};

    CD3DX12_ROOT_PARAMETER rootParameters[12];
    rootParameters[0].InitAsDescriptorTable(9, ranges);
    rootParameters[1].InitAsShaderResourceView(3, 0);  // Geometry Info
    rootParameters[2].InitAsShaderResourceView(4, 0);  // Instance Info
    rootParameters[3].InitAsShaderResourceView(5, 0);  // Random buffer
    rootParameters[4].InitAsShaderResourceView(6, 0);  // Lights struct buffer
    rootParameters[5].InitAsConstants(1, 1);            // Time
    rootParameters[6].InitAsConstantBufferView(3, 0);   // Pass constants
    rootParameters[7].InitAsConstantBufferView(4, 0);   // Voxel grid constants
    rootParameters[8].InitAsUnorderedAccessView(3, 0);  // Guiding counters
    rootParameters[9].InitAsUnorderedAccessView(4, 0);  // Guiding compact ids
    rootParameters[10].InitAsUnorderedAccessView(5, 0); // Guiding CDF
    rootParameters[11].InitAsUnorderedAccessView(6, 0); // Guiding inverse index

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(12, rootParameters);

    auto static_samplers = Renderer::GetStaticSamplers();
    rootSignatureDesc.NumStaticSamplers = static_cast<UINT>(static_samplers.size());
    rootSignatureDesc.pStaticSamplers   = static_samplers.data();
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSignature)));
    m_globalRootSignature->SetName(L"GuidedPathTracing GlobalRootSig");
}

void GuidedPathTracingPass::Render()
{
    if (!m_voxelPass || !m_buildPass)
    {
        spdlog::warn("GuidedPathTracingPass: guiding resources not wired, skipping render");
        return;
    }

    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());
    m_commandList->SetGraphicsRootSignature(nullptr);

    std::vector heaps = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(static_cast<uint32_t>(heaps.size()), heaps.data());
    m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
    m_commandList->SetComputeRootShaderResourceView(1, m_currentScene->GetGeometryInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(2, m_currentScene->GetInstanceInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(3, m_randomBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(4, m_currentScene->GetLightDataBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());

    uint32_t time;
    memcpy(&time, &m_time, sizeof(float));
    m_commandList->SetComputeRoot32BitConstant(5, time, 0);
    m_commandList->SetComputeRootConstantBufferView(6, m_passConstants->GetGpuVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(7, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(8, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(9, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(10, m_buildPass->GetCdfBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(11, m_buildPass->GetInverseIndexBuffer()->GetGPUVirtualAddress());

    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputResource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &transition);
    }

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
    m_commandList->DispatchRays(&desc);

    {
        CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_outputResource.Get());
        m_commandList->ResourceBarrier(1, &uavBarrier);
    }

    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputResource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m_commandList->ResourceBarrier(1, &transition);
    }
}

REGISTER_RAYTRACE_TECHNIQUE("Guided Path Tracing (VXPG)", GuidedPathTracingPass)
