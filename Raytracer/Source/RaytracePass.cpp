#include "pch.h"
#include "RaytracePass.h"

#include <wrl/client.h>

#include "AccelerationStructures.h"
#include "Constants.h"
#include "DXRHelper.h"
#include "Renderer.h"
#include "Shader.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"
#include "Resources/ShaderBindingTable.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"


// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

std::vector<TechniqueEntry>& RaytracePass::GetRegistry()
{
    static std::vector<TechniqueEntry> registry;
    return registry;
}

int RaytracePass::RegisterTechnique(const std::string& name, std::function<std::shared_ptr<RaytracePass>()> factory)
{
    GetRegistry().push_back({name, std::move(factory)});
    return static_cast<int>(GetRegistry().size()) - 1;
}


// ---------------------------------------------------------------------------
// Initialize / lifecycle
// ---------------------------------------------------------------------------

void RaytracePass::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                              Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                              std::shared_ptr<Scene> initialScene,
                              Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> cbvSrvUavHeap,
                              Microsoft::WRL::ComPtr<ID3D12Resource> randomBuffer,
                              std::shared_ptr<PassConstants> passConstants)
{
    spdlog::info("Initializing raytracer pass...");

    m_device       = device;
    m_commandList  = commandList;
    m_srvUavHeap   = cbvSrvUavHeap;
    m_currentScene = initialScene;
    m_randomBuffer = randomBuffer;
    m_passConstants = passConstants;

    InitializeRaytracingPipeline();
    CreateRaytracingOutputBuffer();
    CreateShaderResourceHeap();
    CreateShaderBindingTable();

    spdlog::info("Raytracer pass initialized successfully.");
}

void RaytracePass::Update(double /*elapsedTime*/, double totalTime)
{
    m_time = static_cast<float>(totalTime);
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


// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void RaytracePass::Render()
{
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


// ---------------------------------------------------------------------------
// Default TechniqueDesc (original path tracing shaders)
// ---------------------------------------------------------------------------

TechniqueDesc RaytracePass::GetTechniqueDesc() const
{
    // Default implementation mirrors the original hardcoded path tracing setup.
    // Subclasses override this to provide their own shaders and hit groups.
    TechniqueDesc desc;
    desc.shaders = {
        {"resources/shaders/raytracing.rg.shader",         L"RayGen",     ShaderRole::RayGen},
        {"resources/shaders/raytracing.ms.shader",         L"Miss",       ShaderRole::Miss},
        {"resources/shaders/raytracing.ch.shader",         L"Hit",        ShaderRole::ClosestHit},
        {"resources/shaders/raytracing.shadowhit.shader",  L"ShadowHit",  ShaderRole::AnyHit},
        {"resources/shaders/raytracing.shadowmiss.shader", L"ShadowMiss", ShaderRole::Miss},
    };
    desc.hitGroups = {
        {L"PrimaryHitGroup", L"Hit",  L""},
        {L"ShadowHitGroup",  L"",     L"ShadowHit"},
    };
    desc.maxPayloadSize    = 8 * sizeof(float);
    desc.maxAttributeSize  = 2 * sizeof(float);
    desc.maxRecursionDepth = 8;
    return desc;
}


// ---------------------------------------------------------------------------
// Pipeline initialization — generic over TechniqueDesc
// ---------------------------------------------------------------------------

void RaytracePass::LoadShaders()
{
    auto& rm = ResourceManager::Get();
    spdlog::debug("Loading raytracing shaders from TechniqueDesc ({} shaders)", m_techniqueDesc.shaders.size());

    m_shaderBlobs.clear();
    m_shaderBlobs.reserve(m_techniqueDesc.shaders.size());

    for (const auto& sd : m_techniqueDesc.shaders)
    {
        auto handle = rm.GetOrLoadShader(AssetId(sd.shaderPath));
        m_shaderBlobs.push_back(rm.shaders.GetResource(handle).bytecode);
    }
}

void RaytracePass::InitializeRaytracingPipeline()
{
    spdlog::debug("Initializing raytracing pipeline");

    m_techniqueDesc = GetTechniqueDesc();

    // Load + compile all shaders listed in the descriptor
    LoadShaders();

    // Build local and global root signatures (virtual — subclass can override)
    CreateLocalRootSignatures();
    CreateGlobalRootSignature();

    CD3DX12_STATE_OBJECT_DESC raytracingPipeline{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE};

    // 1. DXIL library subobjects — one per shader
    spdlog::debug("Adding {} shader DXIL libraries to pipeline", m_techniqueDesc.shaders.size());
    assert(m_shaderBlobs.size() == m_techniqueDesc.shaders.size());
    for (size_t i = 0; i < m_techniqueDesc.shaders.size(); ++i)
    {
        const auto& sd   = m_techniqueDesc.shaders[i];
        const auto& blob = m_shaderBlobs[i];
        auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
        D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(blob->GetBufferPointer(), blob->GetBufferSize());
        lib->SetDXILLibrary(&libdxil);
        lib->DefineExport(sd.exportName.c_str(), sd.exportName.c_str());
    }

    // 2. Hit groups
    spdlog::debug("Adding {} hit groups to pipeline", m_techniqueDesc.hitGroups.size());
    for (const auto& hg : m_techniqueDesc.hitGroups)
    {
        auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
        hitGroup->SetHitGroupExport(hg.name.c_str());
        hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
        if (!hg.closestHitExport.empty())
            hitGroup->SetClosestHitShaderImport(hg.closestHitExport.c_str());
        if (!hg.anyHitExport.empty())
            hitGroup->SetAnyHitShaderImport(hg.anyHitExport.c_str());
    }

    // 3. Local root signature associations (group shaders by role)
    spdlog::debug("Associating local root signatures");

    // RayGen shaders
    {
        auto localSig = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localSig->SetRootSignature(m_rayGenLocalSig.Get());
        auto assoc = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        assoc->SetSubobjectToAssociate(*localSig);
        for (const auto& sd : m_techniqueDesc.shaders)
            if (sd.role == ShaderRole::RayGen)
                assoc->AddExport(sd.exportName.c_str());
    }

    // Miss shaders
    {
        auto localSig = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localSig->SetRootSignature(m_missLocalSig.Get());
        auto assoc = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        assoc->SetSubobjectToAssociate(*localSig);
        for (const auto& sd : m_techniqueDesc.shaders)
            if (sd.role == ShaderRole::Miss)
                assoc->AddExport(sd.exportName.c_str());
    }

    // Hit groups (associate by group name, not individual shader export name)
    {
        auto localSig = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localSig->SetRootSignature(m_hitLocalSig.Get());
        auto assoc = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        assoc->SetSubobjectToAssociate(*localSig);
        for (const auto& hg : m_techniqueDesc.hitGroups)
            assoc->AddExport(hg.name.c_str());
    }

    // Global root signature
    {
        auto globalSig = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
        globalSig->SetRootSignature(m_globalRootSignature.Get());
    }

    // 4. Shader payload/attribute configuration
    {
        auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
        shaderConfig->Config(m_techniqueDesc.maxPayloadSize, m_techniqueDesc.maxAttributeSize);
    }

    // 5. Pipeline recursion depth
    {
        auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
        pipelineConfig->Config(m_techniqueDesc.maxRecursionDepth);
    }

    ThrowIfFailed(m_device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_rtStateObject)));
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProperties)));
}


// ---------------------------------------------------------------------------
// Default local root signatures — empty, one per role group
// ---------------------------------------------------------------------------

void RaytracePass::CreateLocalRootSignatures()
{
    spdlog::debug("Creating local root signatures (empty defaults)");

    auto createEmptyLocalSig = [&](Microsoft::WRL::ComPtr<ID3D12RootSignature>& outSig)
    {
        CD3DX12_ROOT_SIGNATURE_DESC desc(0, nullptr);
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
        Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
        ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
        ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&outSig)));
    };

    createEmptyLocalSig(m_rayGenLocalSig);
    createEmptyLocalSig(m_missLocalSig);
    createEmptyLocalSig(m_hitLocalSig);
}


// ---------------------------------------------------------------------------
// Default global root signature — standard 7-param scene bindings
// ---------------------------------------------------------------------------

void RaytracePass::CreateGlobalRootSignature()
{
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

    D3D12_DESCRIPTOR_RANGE ranges[7] = {cbvRange, rtRange, tlasRange, vertex_range, index_range, texture_range, skybox_range};

    CD3DX12_ROOT_PARAMETER rootParameters[7];
    rootParameters[0].InitAsDescriptorTable(7, ranges);
    rootParameters[1].InitAsShaderResourceView(3, 0); // Geometry Info
    rootParameters[2].InitAsShaderResourceView(4, 0); // Instance Info
    rootParameters[3].InitAsShaderResourceView(5, 0); // Random buffer
    rootParameters[4].InitAsShaderResourceView(6, 0); // Lights struct buffer
    rootParameters[5].InitAsConstants(1, 1);           // Time
    rootParameters[6].InitAsConstantBufferView(3, 0);  // Pass constants

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(7, rootParameters);

    auto static_samplers = Renderer::GetStaticSamplers();
    rootSignatureDesc.NumStaticSamplers = static_cast<UINT>(static_samplers.size());
    rootSignatureDesc.pStaticSamplers   = static_samplers.data();
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSignature)));
}


// ---------------------------------------------------------------------------
// SBT — generic, built from TechniqueDesc hit groups + shader roles
// ---------------------------------------------------------------------------

void RaytracePass::CreateShaderBindingTable()
{
    spdlog::info("Creating shader binding table");

    SBTDescriptor sbt_desc = {};

    for (const auto& sd : m_techniqueDesc.shaders)
        if (sd.role == ShaderRole::RayGen)
            sbt_desc.RayGenShaders.push_back({sd.exportName, {}});

    for (const auto& sd : m_techniqueDesc.shaders)
        if (sd.role == ShaderRole::Miss)
            sbt_desc.MissShaders.push_back({sd.exportName, {}});

    for (const auto& hg : m_techniqueDesc.hitGroups)
        sbt_desc.HitShaders.push_back({hg.name, {}});

    m_shaderBindingTable = std::make_shared<ShaderBindingTable>(m_device, m_rtStateObjectProperties, sbt_desc);

    spdlog::info("SBT created ({} raygen, {} miss, {} hit groups)",
        sbt_desc.RayGenShaders.size(), sbt_desc.MissShaders.size(), sbt_desc.HitShaders.size());
}


// ---------------------------------------------------------------------------
// Output buffer + TLAS descriptor
// ---------------------------------------------------------------------------

void RaytracePass::CreateRaytracingOutputBuffer()
{
    spdlog::debug("Creating raytracing output buffer");
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

    ThrowIfFailed(m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &outputBufferDesc,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS(&m_outputResource)));
}

void RaytracePass::CreateShaderResourceHeap()
{
    auto increment = m_device->GetDescriptorHandleIncrementSize(m_srvUavHeap->GetDesc().Type);

    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
    srvHandle.ptr = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart().ptr + 2 * increment; // slot 2: UAV

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc, srvHandle);

    srvHandle.ptr += increment; // slot 3: TLAS

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format                                   = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = m_currentScene->GetAccelerationStructures()->GetTopLevelAS().p_result->GetGPUVirtualAddress();
    m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);
}
