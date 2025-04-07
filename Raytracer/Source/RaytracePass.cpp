#include "pch.h"
#include "RaytracePass.h"

#include <wrl/client.h>

#include "AccelerationStructures.h"
#include "DXRHelper.h"
#include "RaytracingPipelineGenerator.h"
#include "ResourceManager.h"
#include "RootSignatureGenerator.h"
#include "Shader.h"
#include "ShaderBindingTableGenerator.h"
#include "Window.h"

void RaytracePass::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                              Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                              std::shared_ptr<AccelerationStructures> accelerationStructures)
{
    m_device = device;
    m_commandList = commandList;
    m_accelerationStructures = accelerationStructures;
    m_shaderBindingTableGenerator = std::make_shared<nv_helpers_dx12::ShaderBindingTableGenerator>();

    InitializeRaytracingPipeline();
    CreateRaytracingOutputBuffer();
    CreateConstantCameraBuffer();
    CreateShaderResourceHeap();
    CreateShaderBindingTable();
}

void RaytracePass::Render(Microsoft::WRL::ComPtr<ID3D12Resource> renderTarget)
{
    spdlog::debug("Executing raytracing pass");
    
    std::vector<ID3D12DescriptorHeap*> heaps = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(static_cast<uint32_t>(heaps.size()), heaps.data());

    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputResource.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_commandList->ResourceBarrier(1, &transition);
    }

    D3D12_DISPATCH_RAYS_DESC desc = {};

    uint32_t rayGenerationSelectionSizeInBytes = m_shaderBindingTableGenerator->GetRayGenSectionSize();
    desc.RayGenerationShaderRecord.StartAddress = m_shaderBindingTableStorage->GetGPUVirtualAddress();
    desc.RayGenerationShaderRecord.SizeInBytes = rayGenerationSelectionSizeInBytes;

    uint32_t missSelectionSizeInBytes = m_shaderBindingTableGenerator->GetMissSectionSize();
    desc.MissShaderTable.StartAddress = m_shaderBindingTableStorage->GetGPUVirtualAddress() + rayGenerationSelectionSizeInBytes;
    desc.MissShaderTable.StrideInBytes = missSelectionSizeInBytes;
    desc.MissShaderTable.SizeInBytes = m_shaderBindingTableGenerator->GetMissEntrySize();

    uint32_t hitGroupsSelectionSize = m_shaderBindingTableGenerator->GetHitGroupSectionSize();
    desc.HitGroupTable.StartAddress = m_shaderBindingTableStorage->GetGPUVirtualAddress() + rayGenerationSelectionSizeInBytes + missSelectionSizeInBytes;
    desc.HitGroupTable.StrideInBytes = hitGroupsSelectionSize;
    desc.HitGroupTable.SizeInBytes = m_shaderBindingTableGenerator->GetHitGroupEntrySize();

    // For some reason this wasn't aligned correctly - ensuring alignment
    desc.HitGroupTable.StartAddress = (desc.HitGroupTable.StartAddress + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1) & ~(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1);
    desc.HitGroupTable.StrideInBytes = (desc.HitGroupTable.StrideInBytes + D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1) & ~(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1); 

    desc.Width = Window::Get().GetWidth();
    desc.Height = Window::Get().GetHeight();
    desc.Depth = 1;

    m_commandList->SetPipelineState1(m_rtStateObject.Get());
    m_commandList->DispatchRays(&desc);

    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputResource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);

        m_commandList->ResourceBarrier(1, &transition);
    }

    {
       CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition( 
            renderTarget.Get(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_DEST);

        m_commandList->ResourceBarrier(1, &transition);
    }

    m_commandList->CopyResource(renderTarget.Get(), m_outputResource.Get());
    
    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            renderTarget.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET);

        m_commandList->ResourceBarrier(1, &transition);
    }
}

void RaytracePass::Update(DirectX::XMMATRIX view, DirectX::XMMATRIX proj)
{
    std::vector<DirectX::XMMATRIX> matrices(4);
    matrices[0] = view;
    matrices[1] = proj;
    DirectX::XMVECTOR det;
    matrices[2] = XMMatrixInverse(&det, view);
    matrices[3] = XMMatrixInverse(&det, proj);

    uint8_t* pData;
    ThrowIfFailed(m_cbCamera->Map(0, nullptr, (void**)&pData));
    memcpy(pData, matrices.data(), sizeof(DirectX::XMMATRIX) * matrices.size());
    m_cbCamera->Unmap(0, nullptr);
}

void RaytracePass::InitializeRaytracingPipeline()
{
    spdlog::debug("Initializing raytracing pipeline");
    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_device.Get());

    auto rm = ResourceManager::Get();

    spdlog::debug("Loading raytracing shaders");
    m_rayGenShaderBlob = nv_helpers_dx12::CompileShaderLibrary(L"resources/shaders/raytracing/raygen.hlsl"); 

    m_missShaderBlob = nv_helpers_dx12::CompileShaderLibrary(L"resources/shaders/raytracing/miss.hlsl");

    m_hitShaderBlob = nv_helpers_dx12::CompileShaderLibrary(L"resources/shaders/raytracing/hit.hlsl");

    spdlog::debug("Adding raytracing shaders to pipeline");
    m_rayGenShaderName = L"RayGen";
    pipeline.AddLibrary(m_rayGenShaderBlob.Get(), {m_rayGenShaderName});

    m_missShaderName = L"Miss";
    pipeline.AddLibrary(m_missShaderBlob.Get(), {m_missShaderName});

    m_hitShaderName = L"ClosestHit";
    pipeline.AddLibrary(m_hitShaderBlob.Get(), {m_hitShaderName});

    spdlog::debug("Creating raytracing root signatures");
    m_rayGenSignature = CreateRayGenSignature();
    m_missSignature = CreateMissSignature();
    m_hitSignature = CreateHitSignature();

    spdlog::debug("Adding hit group to pipeline");
    pipeline.AddHitGroup(L"HitGroup", m_hitShaderName);

    spdlog::debug("Associating root signatures with pipeline");
    pipeline.AddRootSignatureAssociation(m_rayGenSignature.Get(), {m_rayGenShaderName});
    pipeline.AddRootSignatureAssociation(m_missSignature.Get(), {m_missShaderName});
    pipeline.AddRootSignatureAssociation(m_hitSignature.Get(), {m_hitShaderName});

    spdlog::debug("Setting pipeline payload and attributes");
    pipeline.SetMaxPayloadSize(4 * sizeof(float)); // RGB + distance
    pipeline.SetMaxAttributeSize(2 * sizeof(float)); // barycentric coordinates
    pipeline.SetMaxRecursionDepth(1);

    spdlog::debug("Generating raytracing pipeline state object");
    m_rtStateObject = pipeline.Generate();
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProperties)));
}

Microsoft::WRL::ComPtr<ID3D12RootSignature> RaytracePass::CreateRayGenSignature()
{
    spdlog::debug("Creating ray generation root signature");
    nv_helpers_dx12::RootSignatureGenerator rsGenerator;
    
    D3D12_DESCRIPTOR_RANGE outputBufferRange;
    outputBufferRange.BaseShaderRegister = 0;
    outputBufferRange.NumDescriptors = 1;
    outputBufferRange.RegisterSpace = 0;
    outputBufferRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    outputBufferRange.OffsetInDescriptorsFromTableStart = 0;

    D3D12_DESCRIPTOR_RANGE tlasRange;
    tlasRange.BaseShaderRegister = 0;
    tlasRange.NumDescriptors = 1;
    tlasRange.RegisterSpace = 0;
    tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    tlasRange.OffsetInDescriptorsFromTableStart = 1;

    D3D12_DESCRIPTOR_RANGE cbvRange;
    cbvRange.BaseShaderRegister = 0;
    cbvRange.NumDescriptors = 1;
    cbvRange.RegisterSpace = 0;
    cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    cbvRange.OffsetInDescriptorsFromTableStart = 2;
    
    rsGenerator.AddHeapRangesParameter({outputBufferRange, tlasRange, cbvRange});

    spdlog::debug("Generating ray generation root signature");
    return rsGenerator.Generate(m_device.Get(), true);
}

Microsoft::WRL::ComPtr<ID3D12RootSignature> RaytracePass::CreateMissSignature()
{
    spdlog::debug("Creating miss root signature");
    nv_helpers_dx12::RootSignatureGenerator rsGenerator;

    spdlog::debug("Generating miss root signature");
    return rsGenerator.Generate(m_device.Get(), true);
}

Microsoft::WRL::ComPtr<ID3D12RootSignature> RaytracePass::CreateHitSignature()
{
    spdlog::debug("Creating hit root signature");
    nv_helpers_dx12::RootSignatureGenerator rsGenerator;

    spdlog::debug("Generating hit root signature");
    return rsGenerator.Generate(m_device.Get(), true);
}

void RaytracePass::CreateRaytracingOutputBuffer()
{
    spdlog::debug("Creating raytracing output buffer");
    D3D12_RESOURCE_DESC outputBufferDesc = {};
    outputBufferDesc.DepthOrArraySize = 1;
    outputBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    outputBufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    outputBufferDesc.Width = Window::Get().GetWidth();
    outputBufferDesc.Height = Window::Get().GetHeight();
    outputBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    outputBufferDesc.MipLevels = 1;
    outputBufferDesc.SampleDesc.Count = 1;
    outputBufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    spdlog::debug("Creating commited raytracing output buffer resource");
    ThrowIfFailed(m_device->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &outputBufferDesc,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        nullptr,
        IID_PPV_ARGS(&m_outputResource)));
}

void RaytracePass::CreateShaderResourceHeap()
{
    spdlog::debug("Creating SRV UAV shader resource heap for raytracing");
    m_srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(
        m_device.Get(), 3, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

    spdlog::debug("Creating UAV for raytracing output buffer");
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc, srvHandle);

    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = m_accelerationStructures->GetTopLevelAS().p_result->GetGPUVirtualAddress();

    spdlog::debug("Creating SRV for top level acceleration structure");
    m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
    cbvDesc.BufferLocation = m_cbCamera->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = static_cast<UINT>(sizeof(DirectX::XMMATRIX) * 4 + 255) & ~255;

    m_device->CreateConstantBufferView(&cbvDesc, srvHandle);  
}

void RaytracePass::CreateShaderBindingTable()
{
    spdlog::debug("Creating shader binding table");
    m_shaderBindingTableGenerator->Reset();

    D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();

    uint64_t *heapPtr = reinterpret_cast<uint64_t*>(srvUavHeapHandle.ptr);

    spdlog::debug("Adding shaders to shader binding table");
    m_shaderBindingTableGenerator->AddRayGenerationProgram(m_rayGenShaderName, {heapPtr});
    m_shaderBindingTableGenerator->AddMissProgram(m_missShaderName, {});
    m_shaderBindingTableGenerator->AddHitGroup(L"HitGroup", {});

    uint32_t sbtSize = m_shaderBindingTableGenerator->ComputeSBTSize();

    spdlog::debug("Creating shader binding table resource");
    m_shaderBindingTableStorage = nv_helpers_dx12::CreateBuffer(
        m_device.Get(), sbtSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
        nv_helpers_dx12::kUploadHeapProps);
    if (!m_shaderBindingTableStorage)
    {
        throw std::logic_error("Could not allocate shader binding table resource");
    }

    spdlog::debug("Compiling the shader binding table");
    m_shaderBindingTableGenerator->Generate(m_shaderBindingTableStorage.Get(), m_rtStateObjectProperties.Get());
}

void RaytracePass::CreateConstantCameraBuffer()
{
    m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(256),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_cbCamera));

    m_cbCamera->SetName(L"Raytracing camera world view projection buffer");
}
