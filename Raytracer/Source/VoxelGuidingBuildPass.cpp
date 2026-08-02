#include "pch.h"
#include "CommandContext.h"
#include "VoxelGuidingBuildPass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
// The three voxel textures live in this pass's private heap; everything else is
// a root descriptor. All three kernels share the layout.
constexpr BindingSlot kGuidingBuildConstants = RootConstants("BuildCB", GUIDING_BUILD_REG_CB, 4);
constexpr BindingSlot kGuidingBuildIrradiance =
    TableEntryAt("gVoxIrradiance", BindingKind::Uav, GUIDING_BUILD_REG_IRRADIANCE, 0);
constexpr BindingSlot kGuidingBuildVplCount =
    TableEntryAt("gVoxVplCount", BindingKind::Uav, GUIDING_BUILD_REG_VPL_COUNT, 1);
constexpr BindingSlot kGuidingBuildRepresentative =
    TableEntryAt("gVoxelRepresentative", BindingKind::Uav, GUIDING_BUILD_REG_VOXEL_REPRESENTATIVE, 2);
constexpr BindingSlot kGuidingBuildCounters      = RootUav("gCounters", GUIDING_BUILD_REG_COUNTERS);
constexpr BindingSlot kGuidingBuildCompactIds    = RootUav("gCompactIds", GUIDING_BUILD_REG_COMPACT_IDS);
constexpr BindingSlot kGuidingBuildInverseIndex  = RootUav("gInverseIndex", GUIDING_BUILD_REG_INVERSE_INDEX);
constexpr BindingSlot kGuidingBuildLiveBoundMin  = RootUav("gLiveBoundMin", GUIDING_BUILD_REG_LIVE_BOUND_MIN);
constexpr BindingSlot kGuidingBuildLiveBoundMax  = RootUav("gLiveBoundMax", GUIDING_BUILD_REG_LIVE_BOUND_MAX);
constexpr BindingSlot kGuidingBuildLightPoints =
    RootUav("gCompactVoxelLightPoints", GUIDING_BUILD_REG_COMPACT_LIGHT_POINTS);
constexpr BindingSlot kGuidingBuildPremulIrradiance = RootUav("gPremulIrradiance", GUIDING_BUILD_REG_PREMUL_IRRADIANCE);
constexpr BindingSlot kGuidingBuildBakedBoundMin    = RootUav("gBakedBoundMin", GUIDING_BUILD_REG_BAKED_BOUND_MIN);
constexpr BindingSlot kGuidingBuildBakedBoundMax    = RootUav("gBakedBoundMax", GUIDING_BUILD_REG_BAKED_BOUND_MAX);

constexpr BindingSlot kGuidingBuildSlots[] = {
    kGuidingBuildConstants,     kGuidingBuildIrradiance,   kGuidingBuildVplCount,      kGuidingBuildRepresentative,
    kGuidingBuildCounters,      kGuidingBuildCompactIds,   kGuidingBuildInverseIndex,  kGuidingBuildLiveBoundMin,
    kGuidingBuildLiveBoundMax,  kGuidingBuildLightPoints,  kGuidingBuildPremulIrradiance,
    kGuidingBuildBakedBoundMin, kGuidingBuildBakedBoundMax};
} // namespace

void VoxelGuidingBuildPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>  voxelPass)
{
    spdlog::info("Initializing voxel guiding build pass...");

    m_device      = device;
    m_commandList = commandList;
    m_voxelPass   = std::move(voxelPass);

    CreateBuffers();
    CreateDescriptorHeap();
    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void VoxelGuidingBuildPass::CreateBuffers()
{
    constexpr uint32_t capacity = Constants::Graphics::VOXEL_GUIDING_CAPACITY;

    // Counters buffer stays 2 elements: element [1] is retired (was the CDF
    // total weight) but downstream root-UAV consumers still see stride 2.
    m_counters     = std::make_unique<RWStructuredBuffer<uint32_t>>(m_device, 2,        L"VoxelGuiding Counters");
    m_compactIds   = std::make_unique<RWStructuredBuffer<uint32_t>>(m_device, capacity, L"VoxelGuiding CompactIds");
    m_compactVoxelLightPoints = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT4>>(m_device, capacity, L"VoxelGuiding CompactVoxelLightPoints");
    m_premulIrradiance = std::make_unique<RWStructuredBuffer<float>>(m_device, capacity, L"VoxelGuiding PremulIrradiance");

    CreateGridSizedBuffers();
}

void VoxelGuidingBuildPass::CreateGridSizedBuffers()
{
    // Grid-sized (one element per cell), not capacity-sized. Bound as ROOT
    // UAVs (no bounds checking), so they MUST track the grid dim exactly —
    // undersized means the shaders write past the end and corrupt GPU memory.
    const uint32_t gridDim = m_voxelPass->GetGridDim();
    const uint32_t cellCount = gridDim * gridDim * gridDim;
    m_inverseIndex = std::make_unique<RWStructuredBuffer<int32_t>>(m_device, cellCount, L"VoxelGuiding InverseIndex");
    m_liveBoundMin = std::make_unique<RWStructuredBuffer<DirectX::XMUINT4>>(m_device, cellCount, L"VoxelGuiding LiveBoundMin");
    m_liveBoundMax = std::make_unique<RWStructuredBuffer<DirectX::XMUINT4>>(m_device, cellCount, L"VoxelGuiding LiveBoundMax");
}

void VoxelGuidingBuildPass::OnVoxelGridResize()
{
    if (!m_initialized)
        return;
    CreateGridSizedBuffers();
    // Force a descriptor rebind: the voxel textures were recreated too.
    m_boundIrradiance = nullptr;
    m_boundVplCount = nullptr;
    m_boundRepresentative = nullptr;
}

void VoxelGuidingBuildPass::CreateDescriptorHeap()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 3; // irradiance + vpl count + representative VPL
    heapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_descHeap)));
    m_descHeap->SetName(L"VoxelGuidingBuild Heap");
}

void VoxelGuidingBuildPass::RebindDescriptorsIfChanged(ID3D12Resource* representativeTex)
{
    ID3D12Resource* irradiance = m_voxelPass->GetIrradianceTexture().Get();
    ID3D12Resource* vplCount   = m_voxelPass->GetVplCountTexture().Get();
    if (irradiance == m_boundIrradiance && vplCount == m_boundVplCount &&
        representativeTex == m_boundRepresentative)
        return;

    UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_descHeap->GetCPUDescriptorHandleForHeapStart());

    m_voxelPass->WriteIrradianceUavTo(handle);
    handle.Offset(1, inc);
    m_voxelPass->WriteVplCountUavTo(handle);
    handle.Offset(1, inc);

    if (representativeTex)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format          = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.ViewDimension   = D3D12_UAV_DIMENSION_TEXTURE3D;
        uavDesc.Texture3D.WSize = m_voxelPass->GetGridDim();
        m_device->CreateUnorderedAccessView(representativeTex, nullptr, &uavDesc, handle);
    }

    m_boundIrradiance     = irradiance;
    m_boundVplCount       = vplCount;
    m_boundRepresentative = representativeTex;
}

void VoxelGuidingBuildPass::CreateRootSignature()
{
    m_rootSig = RootSignatureBuilder(L"VoxelGuidingBuild RootSig", /*tableCount*/ 1)
                    .Add(kGuidingBuildSlots)
                    .Build(m_device.Get());
}

void VoxelGuidingBuildPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/voxelGuidingBuild.clear.shader", L"VoxelGuiding Clear PSO");
    m_reloadProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/voxelGuidingBuild.reload.shader", L"VoxelGuiding Reload PSO");
    m_compactProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/voxelGuidingBuild.compact.shader", L"VoxelGuiding Compact PSO");
}

bool VoxelGuidingBuildPass::BindCommon()
{
    if (!m_initialized || !m_voxelPass)
        return false;

    RebindDescriptorsIfChanged(m_representativeTex);

    const uint32_t gridDim = m_voxelPass->GetGridDim();

    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSig.Get());

    uint32_t constants[4] = { gridDim, 0, 0, 0 };
    auto* cmd = m_commandList.Get();
    m_rootSig.SetConstants(cmd, kGuidingBuildConstants, constants, 4);
    m_rootSig.SetTable(cmd, 0, m_descHeap->GetGPUDescriptorHandleForHeapStart());
    m_rootSig.Set(cmd, kGuidingBuildCounters, m_counters->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildCompactIds, m_compactIds->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildInverseIndex, m_inverseIndex->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildLiveBoundMin, m_liveBoundMin->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildLiveBoundMax, m_liveBoundMax->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildLightPoints, m_compactVoxelLightPoints->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildPremulIrradiance, m_premulIrradiance->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildBakedBoundMin, m_voxelPass->GetBakedBoundMinBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kGuidingBuildBakedBoundMax, m_voxelPass->GetBakedBoundMaxBuffer()->GetGPUVirtualAddress());

    return true;
}

void VoxelGuidingBuildPass::RunClear()
{
    if (!BindCommon())
        return;

    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());
    CommandContext::Get().Dispatch(1, 1, 1);
}

// Reload baked bounds for lit voxels before compaction reads them.
void VoxelGuidingBuildPass::RunReload()
{
    if (!BindCommon())
        return;

    const uint32_t groups = (m_voxelPass->GetGridDim() + 7) / 8;
    m_commandList->SetPipelineState(m_reloadProgram->GetPipelineState());
    CommandContext::Get().Dispatch(groups, groups, groups);
}

void VoxelGuidingBuildPass::RunCompact()
{
    if (!BindCommon())
        return;

    const uint32_t groups = (m_voxelPass->GetGridDim() + 7) / 8;
    m_commandList->SetPipelineState(m_compactProgram->GetPipelineState());
    CommandContext::Get().Dispatch(groups, groups, groups);
}
