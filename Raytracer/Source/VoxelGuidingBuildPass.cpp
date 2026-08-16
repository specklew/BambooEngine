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
// The three voxel textures are read out of the global heap, at the same slots
// their producers (voxelization, light injection) write; everything else is a
// root descriptor. All three kernels share the layout.
constexpr BindingSlot kGuidingBuildConstants = PassRootConstants("BuildCB", GUIDING_BUILD_REG_CB, 4);
constexpr BindingSlot kGuidingBuildIrradiance =
    PassTableEntry("gVoxIrradiance", BindingKind::Uav, GUIDING_BUILD_REG_IRRADIANCE, GlobalDescriptor::VoxelIrradiance);
constexpr BindingSlot kGuidingBuildVplCount =
    PassTableEntry("gVoxVplCount", BindingKind::Uav, GUIDING_BUILD_REG_VPL_COUNT, GlobalDescriptor::VoxelVplCount);
constexpr BindingSlot kGuidingBuildRepresentative =
    PassTableEntry("gVoxelRepresentative", BindingKind::Uav, GUIDING_BUILD_REG_VOXEL_REPRESENTATIVE,
                   GlobalDescriptor::VoxelRepresentative);
constexpr BindingSlot kGuidingBuildCounters      = PassUav("gCounters", GUIDING_BUILD_REG_COUNTERS);
constexpr BindingSlot kGuidingBuildCompactIds    = PassUav("gCompactIds", GUIDING_BUILD_REG_COMPACT_IDS);
constexpr BindingSlot kGuidingBuildInverseIndex  = PassUav("gInverseIndex", GUIDING_BUILD_REG_INVERSE_INDEX);
constexpr BindingSlot kGuidingBuildLiveBoundMin  = PassUav("gLiveBoundMin", GUIDING_BUILD_REG_LIVE_BOUND_MIN);
constexpr BindingSlot kGuidingBuildLiveBoundMax  = PassUav("gLiveBoundMax", GUIDING_BUILD_REG_LIVE_BOUND_MAX);
constexpr BindingSlot kGuidingBuildLightPoints =
    PassUav("gCompactVoxelLightPoints", GUIDING_BUILD_REG_COMPACT_LIGHT_POINTS);
constexpr BindingSlot kGuidingBuildPremulIrradiance = PassUav("gPremulIrradiance", GUIDING_BUILD_REG_PREMUL_IRRADIANCE);
constexpr BindingSlot kGuidingBuildBakedBoundMin    = PassUav("gBakedBoundMin", GUIDING_BUILD_REG_BAKED_BOUND_MIN);
constexpr BindingSlot kGuidingBuildBakedBoundMax    = PassUav("gBakedBoundMax", GUIDING_BUILD_REG_BAKED_BOUND_MAX);

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

    const uint32_t gridDim = m_voxelPass->GetGridDim();

    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSig.Get());

    uint32_t constants[4] = { gridDim, 0, 0, 0 };
    auto* cmd = m_commandList.Get();
    m_rootSig.SetConstants(cmd, kGuidingBuildConstants, constants, 4);
    m_rootSig.SetTable(cmd, 0, globalHeap.GpuStart());
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
