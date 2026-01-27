#include "pch.h"
#include "RaytracePass.h"

#include <wrl/client.h>

#include "AccelerationStructures.h"
#include "DXRHelper.h"
#include "Shader.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"
#include "Resources/ShaderBindingTable.h"
#include "SceneResources/Scene.h"

void RaytracePass::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                              Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                              std::shared_ptr<Scene> initialScene,
                              Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> cbvSrvUavHeap)
{
    spdlog::info("Initializing raytracer pass...");
    
    m_device = device;
    m_commandList = commandList;
    m_srvUavHeap = cbvSrvUavHeap;
    m_currentScene = initialScene;

    InitializeRaytracingPipeline();
    CreateRaytracingOutputBuffer();
    CreateShaderResourceHeap();
    CreateShaderBindingTable();
    
    spdlog::info("Raytracer pass initialized successfully.");
}

void RaytracePass::Render(const Microsoft::WRL::ComPtr<ID3D12Resource>& renderTarget)
{
    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());
    m_commandList->SetGraphicsRootSignature(nullptr);
    
    std::vector heaps = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(static_cast<uint32_t>(heaps.size()), heaps.data());
    m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

    {
        CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputResource.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_commandList->ResourceBarrier(1, &transition);
    }

    D3D12_DISPATCH_RAYS_DESC desc = {};
    
    desc.RayGenerationShaderRecord.StartAddress = m_shaderBindingTable->GetUnderlyingResource()->GetGPUVirtualAddress();
    desc.RayGenerationShaderRecord.SizeInBytes = m_shaderBindingTable->GetRayGenEntrySize();
    
    desc.MissShaderTable.StartAddress = desc.RayGenerationShaderRecord.StartAddress + desc.RayGenerationShaderRecord.SizeInBytes;
    desc.MissShaderTable.StrideInBytes = m_shaderBindingTable->GetMissSectionSize();
    desc.MissShaderTable.SizeInBytes = m_shaderBindingTable->GetMissEntrySize();
    
    desc.HitGroupTable.StartAddress = desc.MissShaderTable.StartAddress + desc.MissShaderTable.StrideInBytes;
    desc.HitGroupTable.StrideInBytes = m_shaderBindingTable->GetHitSectionSize();
    desc.HitGroupTable.SizeInBytes = m_shaderBindingTable->GetHitEntrySize();

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

void RaytracePass::OnResize()
{
    m_outputResource.Reset();
    CreateRaytracingOutputBuffer();
    CreateShaderResourceHeap();
}

void RaytracePass::OnShaderReload()
{
    InitializeRaytracingPipeline();
    CreateShaderBindingTable();
}

void RaytracePass::OnSceneChange(std::shared_ptr<Scene> scene)
{
    spdlog::debug("Scene change for Ray Tracing...");
    m_currentScene = scene;
    CreateShaderResourceHeap();
}

void RaytracePass::InitializeRaytracingPipeline()
{
    spdlog::debug("Initializing raytracing pipeline");
    CD3DX12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };
    
    // Creating raytracing pipeline state object
    // It is consisting of several subobjects, each defining a part of the pipeline
    // 1. Ray generation shader DXIL
    // 2. Miss shader DXIL
    // 3. Hit shader DXIL
    // 4. Hit group
    // 5. Root signatures for each shader (local root signatures)
    // 6. Shader configuration (payload size, attribute size)
    // 7. Pipeline configuration (max recursion depth)
    
    CreateRootSignatures();
    
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
        pipelineConfig->Config(1); // Max recursion depth TODO: CHANGE THIS WHEN IMPLEMENTING RECURSIVE RT
    }

    ThrowIfFailed(m_device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_rtStateObject)));
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProperties)));
}

void RaytracePass::CreateRootSignatures()
{
    auto& rm = ResourceManager::Get();

    spdlog::debug("Loading raytracing shaders");

    auto rayGenShader = rm.GetOrLoadShader(AssetId("resources/shaders/raytracing.rg.shader"));
    m_rayGenShaderBlob = rm.shaders.GetResource(rayGenShader).bytecode;
    auto missShader = rm.GetOrLoadShader(AssetId("resources/shaders/raytracing.ms.shader"));
    m_missShaderBlob = rm.shaders.GetResource(missShader).bytecode;
    auto hitShader = rm.GetOrLoadShader(AssetId("resources/shaders/raytracing.ch.shader"));
    m_hitShaderBlob = rm.shaders.GetResource(hitShader).bytecode;

    spdlog::debug("Creating raytracing root signatures");
    CreateRayGenSignature();
    CreateMissSignature();
    CreateHitSignature();
    CreateGlobalRootSignature();
}

void RaytracePass::CreateRayGenSignature()
{
    spdlog::debug("Creating ray generation root signature");
    
    CD3DX12_ROOT_SIGNATURE_DESC localRayGenSignatureDesc(0, nullptr);
    localRayGenSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error;

    ThrowIfFailed(D3D12SerializeRootSignature(&localRayGenSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_rayGenSignature)));
}

void RaytracePass::CreateMissSignature()
{
    spdlog::debug("Creating miss root signature");
    
    CD3DX12_ROOT_SIGNATURE_DESC localMissSignatureDesc(0, nullptr);
    localMissSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error;

    ThrowIfFailed(D3D12SerializeRootSignature(&localMissSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_missSignature)));
}

void RaytracePass::CreateHitSignature()
{
    spdlog::debug("Creating hit root signature");

    CD3DX12_ROOT_SIGNATURE_DESC localHitSignatureDesc(0, nullptr);
    localHitSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error;

    ThrowIfFailed(D3D12SerializeRootSignature(&localHitSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_hitSignature)));
}

void RaytracePass::CreateGlobalRootSignature()
{
    CD3DX12_ROOT_PARAMETER rootParameters[1];

    // CBV for ImGui
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

    D3D12_DESCRIPTOR_RANGE ranges[5] = {cbvRange, rtRange, tlasRange, vertex_range, index_range};

    rootParameters[0].InitAsDescriptorTable(5, ranges);
    
    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(1, rootParameters);
    
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
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
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
    srvHandle.ptr = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart().ptr + 2 * increment; // IMGUI | World CBV | UAV <- ptr
    
    spdlog::debug("Creating UAV for raytracing output buffer");
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc, srvHandle);
    
    srvHandle.ptr += increment;
        
    spdlog::debug("Creating SRV for top level acceleration structure");
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = m_currentScene->GetAccelerationStructures()->GetTopLevelAS().p_result->GetGPUVirtualAddress();
    m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

    // CBV for camera is already created in CreateConstantCameraBuffer() in Renderer
}

void RaytracePass::CreateShaderBindingTable()
{
    spdlog::info("Creating shader binding table");
    
    D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    void* heapPtr = reinterpret_cast<void*>(srvUavHeapHandle.ptr);
    
    SBTDescriptor sbt_desc = {};
    sbt_desc.RayGenShaders.push_back({m_rayGenShaderName, {heapPtr}});
    sbt_desc.MissShaders.push_back({m_missShaderName, {}});
    sbt_desc.HitShaders.push_back({m_hitGroupName, {}});

    m_shaderBindingTable = std::make_shared<ShaderBindingTable>(m_device, m_rtStateObjectProperties, sbt_desc);

    spdlog::info("SBT Buffer has been populated with SBT entries successfully");
}
