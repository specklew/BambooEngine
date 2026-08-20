#include "pch.h"
#include "CommandContext.h"
#include "DxrPass.h"

#include <wrl/client.h>

#include "AccelerationStructures.h"
#include "Constants.h"
#include "VendorLevers.h"
#include "DXRHelper.h"
#include "FrameBindingLayout.h"
#include "GlobalDescriptorHeap.h"
#include "PassRegisters.h"
#include "Renderer.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "ShaderReflection.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"
#include "Resources/ShaderBindingTable.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"


// ---------------------------------------------------------------------------
// Initialize / lifecycle
// ---------------------------------------------------------------------------

void DxrPass::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                         Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
                         std::shared_ptr<Scene> initialScene,
                         std::shared_ptr<PassConstants> passConstants)
{
    spdlog::info("Initializing raytracer pass...");

    m_device       = device;
    m_commandList  = commandList;
    m_currentScene = initialScene;
    m_passConstants = passConstants;

    InitializeRaytracingPipeline();
    CreateShaderResourceHeap();
    CreateShaderBindingTable();

    spdlog::info("Raytracer pass initialized successfully.");
}

void DxrPass::OnResize()
{
    CreateShaderResourceHeap();
}

void DxrPass::OnShaderReload()
{
    InitializeRaytracingPipeline();
    CreateShaderBindingTable();
}

void DxrPass::OnSceneChange(std::shared_ptr<Scene> scene)
{
    spdlog::debug("Scene change for Ray Tracing...");
    m_currentScene = scene;
    CreateShaderResourceHeap();
}


// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void DxrPass::GetLaunchExtent(uint32_t& width, uint32_t& height) const
{
    width  = Window::Get().GetWidth();
    height = Window::Get().GetHeight();

    if (!CompilesLever("swizzle"))
        return;

    constexpr uint32_t tile = RAYGEN_SWIZZLE_TILE_SIZE;
    width  = ((width  + tile - 1) / tile) * tile;
    height = ((height + tile - 1) / tile) * tile;
}

void DxrPass::Render()
{
    m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());
    m_commandList->SetGraphicsRootSignature(nullptr);

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

    GetLaunchExtent(desc.Width, desc.Height);
    desc.Depth  = 1;

    m_commandList->SetPipelineState1(m_rtStateObject.Get());
    CommandContext::Get().DispatchRays(desc);
}


// ---------------------------------------------------------------------------
// Default TechniqueDesc (original path tracing shaders)
// ---------------------------------------------------------------------------

TechniqueDesc DxrPass::GetTechniqueDesc() const
{
    // Default implementation mirrors the original hardcoded path tracing setup.
    // Subclasses override this to provide their own shaders and hit groups.
    TechniqueDesc desc;
    desc.shaders = {
        {"resources/shaders/raytracing.rg.shader",          L"RayGen",     ShaderRole::RayGen},
        {"resources/shaders/raytracing.ms.shader",          L"Miss",       ShaderRole::Miss},
        {"resources/shaders/raytracing.ch.shader",          L"Hit",        ShaderRole::ClosestHit},
        {"resources/shaders/raytracing.ah.shader",          L"AnyHit",     ShaderRole::AnyHit},
        {"resources/shaders/raytracing.shadowhit.shader",   L"ShadowHit",  ShaderRole::AnyHit},
        {"resources/shaders/raytracing.shadowmiss.shader",  L"ShadowMiss", ShaderRole::Miss},
    };
    desc.hitGroups = {
        {L"PrimaryHitGroup", L"Hit",  L"AnyHit"},
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

void DxrPass::LoadShaders()
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

// Every library in a state object is served by the one global root signature, so
// each is checked against it and their coverage accumulates.
void DxrPass::ValidateShaderBindings() const
{
    for (const auto& shaderDesc : m_techniqueDesc.shaders)
        ShaderReflection::ValidateLibraryAsset(shaderDesc.shaderPath.c_str(), m_globalRootSignature.Get());
}

void DxrPass::InitializeRaytracingPipeline()
{
    spdlog::debug("Initializing raytracing pipeline");

    m_techniqueDesc = GetTechniqueDesc();

    // Compile-time vendor levers rewrite the RAYGEN asset: that is the kernel whose
    // codegen shape moves 6-11 % from changes this size (ADR 0014/0015), and the
    // hit/miss shaders carry none of the code most levers gate. The exception is a
    // lever that changes a declaration the libraries SHARE — payload qualifiers —
    // which every shader in the state object must be built with or they disagree
    // about the payload they hand each other. Applied here rather than in each
    // technique's descriptor so a new technique inherits it.
    const std::string sharedKey = VendorLevers::AllShaderSubsetKey(m_shaderVariantKey);
    for (ShaderDesc& shader : m_techniqueDesc.shaders)
    {
        const std::string& key = shader.role == ShaderRole::RayGen ? m_shaderVariantKey : sharedKey;
        shader.shaderPath = VendorLevers::VariantAsset(shader.shaderPath, key);
    }

    // Load + compile all shaders listed in the descriptor
    LoadShaders();

    // Build local and global root signatures (virtual — subclass can override)
    CreateLocalRootSignatures();
    CreateGlobalRootSignature();

    ValidateShaderBindings();

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

void DxrPass::CreateLocalRootSignatures()
{
    spdlog::debug("Creating local root signatures (empty defaults)");

    auto createEmptyLocalSig = [&](Microsoft::WRL::ComPtr<ID3D12RootSignature>& outSig, const wchar_t* debugName)
    {
        CD3DX12_ROOT_SIGNATURE_DESC desc(0, nullptr);
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
        outSig     = RootSignatureLibrary::Get().Create(m_device.Get(), desc, debugName);
    };

    createEmptyLocalSig(m_rayGenLocalSig, L"RayGen LocalRootSig");
    createEmptyLocalSig(m_missLocalSig, L"Miss LocalRootSig");
    createEmptyLocalSig(m_hitLocalSig, L"Hit LocalRootSig");
}


// ---------------------------------------------------------------------------
// Default global root signature — standard 7-param scene bindings
// ---------------------------------------------------------------------------

void DxrPass::CreateGlobalRootSignature()
{
    m_globalRootSignature = RootSignatureBuilder(L"DxrPass GlobalRootSig", /*tableCount*/ 1)
                                .AddFrameLayout()
                                .WithStaticSamplers()
                                .Build(m_device.Get());
}


// ---------------------------------------------------------------------------
// SBT — generic, built from TechniqueDesc hit groups + shader roles
// ---------------------------------------------------------------------------

void DxrPass::CreateShaderBindingTable()
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
// TLAS descriptor — the one view every DXR pass needs
// ---------------------------------------------------------------------------

void DxrPass::CreateShaderResourceHeap()
{
    const D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::Tlas);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format                                   = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = m_currentScene->GetAccelerationStructures()->GetTopLevelAS().p_result->GetGPUVirtualAddress();
    m_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);
}
