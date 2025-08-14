#include "pch.h"
#include "RaytracePass.h"

#include <wrl/client.h>

#include "AccelerationStructures.h"
#include "DXRHelper.h"
#include "RootSignatureGenerator.h"
#include "ShaderBindingTableGenerator.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"

void RaytracePass::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                              Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                              std::shared_ptr<AccelerationStructures> accelerationStructures,
                              Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer,
                              Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer,
                              Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> cbvSrvUavHeap)
{
    spdlog::info("Initializing raytracer pass...");
    
    m_device = device;
    m_commandList = commandList;
    m_accelerationStructures = accelerationStructures;
    m_shaderBindingTableGenerator = std::make_shared<nv_helpers_dx12::ShaderBindingTableGenerator>();
    m_vertexBuffer = vertexBuffer;
    m_indexBuffer = indexBuffer;
    m_srvUavHeap = cbvSrvUavHeap;

    InitializeRaytracingPipeline();
    CreateRaytracingOutputBuffer();
    CreateShaderResourceHeap();
    CreateShaderBindingTable();
    
    spdlog::info("Raytracer pass initialized successfully.");
}

void RaytracePass::Render(const Microsoft::WRL::ComPtr<ID3D12Resource>& renderTarget)
{
    spdlog::debug("Executing raytracing pass");

    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());
    
    std::vector heaps = {m_srvUavHeap.Get()};
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
    desc.MissShaderTable.StartAddress = desc.RayGenerationShaderRecord.StartAddress + desc.RayGenerationShaderRecord.SizeInBytes;
    desc.MissShaderTable.StrideInBytes = missSelectionSizeInBytes;
    desc.MissShaderTable.SizeInBytes = m_shaderBindingTableGenerator->GetMissEntrySize();

    // Start addresses must be aligned to D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT
    // and strides (basically shader record sizes) must be aligned to D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT
    desc.MissShaderTable.StartAddress = (desc.MissShaderTable.StartAddress + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1) & ~(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1);
    desc.MissShaderTable.StrideInBytes = (desc.MissShaderTable.StrideInBytes + D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1) & ~(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1); 
    
    uint32_t hitGroupsSelectionSize = m_shaderBindingTableGenerator->GetHitGroupSectionSize();
    desc.HitGroupTable.StartAddress = desc.MissShaderTable.StartAddress + desc.MissShaderTable.StrideInBytes;
    desc.HitGroupTable.StrideInBytes = hitGroupsSelectionSize;
    desc.HitGroupTable.SizeInBytes = m_shaderBindingTableGenerator->GetHitGroupEntrySize();
    
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
}

void RaytracePass::InitializeRaytracingPipeline()
{
    spdlog::debug("Initializing raytracing pipeline");
    //nv_helpers_dx12::RayTracingPipelineGenerator pipeline(m_device.Get());
    CD3DX12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };

    auto rm = ResourceManager::Get();

    spdlog::debug("Loading raytracing shaders");
    m_rayGenShaderBlob = nv_helpers_dx12::CompileShaderLibrary(L"resources/shaders/raytracing/raygen.hlsl"); 
    m_missShaderBlob = nv_helpers_dx12::CompileShaderLibrary(L"resources/shaders/raytracing/miss.hlsl");
    m_hitShaderBlob = nv_helpers_dx12::CompileShaderLibrary(L"resources/shaders/raytracing/closesthit.hlsl");

    // Creating raytracing pipeline state object
    // It is consisting of several subobjects, each defining a part of the pipeline
    // 1. Ray generation shader DXIL
    // 2. Miss shader DXIL
    // 3. Hit shader DXIL
    // 4. Hit group
    // 5. Root signatures for each shader (local root signatures)
    // 6. Shader configuration (payload size, attribute size)
    // 7. Pipeline configuration (max recursion depth)
    

    spdlog::debug("Adding raytracing shaders DXIL to pipeline");

    // 1. Ray generation shader DXIL
    m_rayGenShaderName = L"RayGen";
    {
        auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
        D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(m_rayGenShaderBlob.Get()->GetBufferPointer(), m_rayGenShaderBlob.Get()->GetBufferSize());
        lib->SetDXILLibrary(&libdxil);
        lib->DefineExport(m_rayGenShaderName.c_str(), m_rayGenShaderName.c_str());
    }

    // 2. Miss shader DXIL
    m_missShaderName = L"Miss";
    {
        auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
        D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(m_missShaderBlob->GetBufferPointer(), m_missShaderBlob->GetBufferSize());
        lib->SetDXILLibrary(&libdxil);
        lib->DefineExport(m_missShaderName.c_str(), m_missShaderName.c_str());
    }

    // 3. Hit shader DXIL
    m_hitShaderName = L"Hit";
    {
        auto lib =  raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
        D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(m_hitShaderBlob->GetBufferPointer(), m_hitShaderBlob->GetBufferSize());
        lib->SetDXILLibrary(&libdxil);
        lib->DefineExport(m_hitShaderName.c_str(), m_hitShaderName.c_str());
    }

    // 4. Hit group
    spdlog::debug("Adding hit group to pipeline");
    m_hitGroupName = L"HitGroup";
    {
        auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
        hitGroup->SetClosestHitShaderImport(m_hitShaderName.c_str());
        hitGroup->SetHitGroupExport(m_hitGroupName.c_str());
        hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
    }
    
    spdlog::debug("Creating raytracing root signatures");
    m_rayGenSignature = CreateRayGenSignature();
    m_missSignature = CreateMissSignature();
    /*m_hitSignature = */CreateHitSignature();
    CreateGloablRootSignature();
    
    // 5. Root signature associations
    spdlog::debug("Associating root signatures with pipeline");
    {
        auto localRootSignatureSubObj = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localRootSignatureSubObj->SetRootSignature(m_rayGenSignature.Get());

        auto rootSignatureAssociation = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignatureSubObj);
        rootSignatureAssociation->AddExport(m_rayGenShaderName.c_str());
    }

    {
        auto localRootSignatureSubObj = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localRootSignatureSubObj->SetRootSignature(m_missSignature.Get());
        
        auto rootSignatureAssociation = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignatureSubObj);
        rootSignatureAssociation->AddExport(m_missShaderName.c_str());
    }

    {
        auto localRootSignatureSubObj = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localRootSignatureSubObj->SetRootSignature(m_hitSignature.Get());

        auto rootSignatureAssociation = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignatureSubObj);
        rootSignatureAssociation->AddExport(m_hitShaderName.c_str());
    }

    {
        auto globalRootSignatureSubObj = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
        globalRootSignatureSubObj->SetRootSignature(m_globalRootSignature.Get());
    }

    // 6. Shader configuration
    spdlog::debug("Setting raytracing shader configuration");
    {
        auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
        uint32_t maxPayloadSize = 4 * sizeof(float); // RGB + distance
        uint32_t maxAttributeSize = 2 * sizeof(float); // barycentric coordinates
        shaderConfig->Config(maxPayloadSize, maxAttributeSize);
    }

    // 7. Pipeline configuration
    spdlog::debug("Setting raytracing pipeline configuration");
    {
        auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
        pipelineConfig->Config(1); // Max recursion depth
    }

    ThrowIfFailed(m_device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_rtStateObject)));
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProperties)));
}

Microsoft::WRL::ComPtr<ID3D12RootSignature> RaytracePass::CreateRayGenSignature()
{
    spdlog::debug("Creating ray generation root signature");
    nv_helpers_dx12::RootSignatureGenerator rsGenerator;

    D3D12_DESCRIPTOR_RANGE cbvRange;
    cbvRange.BaseShaderRegister = 0;
    cbvRange.NumDescriptors = 1;
    cbvRange.RegisterSpace = 0;
    cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    cbvRange.OffsetInDescriptorsFromTableStart = 1;
    
    D3D12_DESCRIPTOR_RANGE outputBufferRange;
    outputBufferRange.BaseShaderRegister = 0;
    outputBufferRange.NumDescriptors = 1;
    outputBufferRange.RegisterSpace = 0;
    outputBufferRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    outputBufferRange.OffsetInDescriptorsFromTableStart = 2;

    D3D12_DESCRIPTOR_RANGE tlasRange;
    tlasRange.BaseShaderRegister = 0;
    tlasRange.NumDescriptors = 1;
    tlasRange.RegisterSpace = 0;
    tlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    tlasRange.OffsetInDescriptorsFromTableStart = 3;
    
    rsGenerator.AddHeapRangesParameter({cbvRange, outputBufferRange, tlasRange});

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

    CD3DX12_ROOT_SIGNATURE_DESC localHitSignatureDesc(0, nullptr);
    localHitSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error;

    ThrowIfFailed(D3D12SerializeRootSignature(&localHitSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_hitSignature)));

    return nullptr;
}

void RaytracePass::CreateGloablRootSignature()
{
    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(0, nullptr);
    
    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSignature)));
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
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;

    auto increment = m_device->GetDescriptorHandleIncrementSize(m_srvUavHeap->GetDesc().Type);
    srvHandle.ptr = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart().ptr + 2 * increment; // IMGUI | CBV | UAV <- ptr
    
    spdlog::debug("Creating UAV for raytracing output buffer");
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc, srvHandle);

    srvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    spdlog::debug("Creating SRV for top level acceleration structure");
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = m_accelerationStructures->GetTopLevelAS().p_result->GetGPUVirtualAddress();
    m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

    // CBV for camera is already created in CreateConstantCameraBuffer() in Renderer
}

void RaytracePass::CreateShaderBindingTable()
{
    spdlog::debug("Creating shader binding table");
    m_shaderBindingTableGenerator->Reset();

    D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();

    void* heapPtr = reinterpret_cast<void*>(srvUavHeapHandle.ptr);

    spdlog::debug("Adding shaders to shader binding table");
    m_shaderBindingTableGenerator->AddRayGenerationProgram(m_rayGenShaderName, {heapPtr});
    m_shaderBindingTableGenerator->AddMissProgram(m_missShaderName, {});
    m_shaderBindingTableGenerator->AddHitGroup(m_hitGroupName, {});
    
    /*reinterpret_cast<void*>(m_vertexBuffer->GetGPUVirtualAddress()), reinterpret_cast<void*>(m_indexBuffer->GetGPUVirtualAddress())*/
    
    uint32_t sbtSize = m_shaderBindingTableGenerator->ComputeSBTSize();

    spdlog::debug("Creating shader binding table resource");
    m_shaderBindingTableStorage = nv_helpers_dx12::CreateBuffer(
        m_device.Get(),
        sbtSize,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nv_helpers_dx12::kUploadHeapProps);
    
    if (!m_shaderBindingTableStorage)
    {
        throw std::logic_error("Could not allocate shader binding table resource");
    }

    spdlog::debug("Compiling the shader binding table");
    m_shaderBindingTableGenerator->Generate(m_shaderBindingTableStorage.Get(), m_rtStateObjectProperties.Get());
}
