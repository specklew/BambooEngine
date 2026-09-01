#include "pch.h"
#include "Utils/GpuMemoryReport.h"
#include "CommandContext.h"
#include "VxpgLightTreePass.h"

#include "Constants.h"
#include "GlobalDescriptorHeap.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgClusterPass.h"
#include "VxpgClusterVisibilityPass.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/CVars.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
// One layout for the whole build: the bottom-tree kernels use the grid CBV and
// u0-u10, the top-level kernel adds its own constants, the two visibility
// buffers, and the mask texture (a table entry, since texture UAVs cannot be
// root descriptors).
constexpr BindingSlot kTreeGridConstants = PassCbv("LightTreeGridCB", LIGHT_TREE_REG_GRID_CB);
constexpr BindingSlot kTreeSortKeys      = PassUav("gSortKeys", LIGHT_TREE_REG_SORT_KEYS);
constexpr BindingSlot kTreeNodes         = PassUav("gNodes", LIGHT_TREE_REG_NODES);
constexpr BindingSlot kTreeLeafRanges    = PassUav("gLeafRanges", LIGHT_TREE_REG_LEAF_RANGES);
constexpr BindingSlot kTreeCompactToLeaf = PassUav("gCompactToLeaf", LIGHT_TREE_REG_COMPACT_TO_LEAF);
constexpr BindingSlot kTreeClusterRoots  = PassUav("gClusterRoots", LIGHT_TREE_REG_CLUSTER_ROOTS);
constexpr BindingSlot kTreeDispatchArgs  = PassUav("gDispatchArgs", LIGHT_TREE_REG_DISPATCH_ARGS);
constexpr BindingSlot kTreeCompactIds    = PassUav("gCompactIds", LIGHT_TREE_REG_COMPACT_IDS);
constexpr BindingSlot kTreeAssignments   = PassUav("gClusterAssignments", LIGHT_TREE_REG_CLUSTER_ASSIGNMENTS);
constexpr BindingSlot kTreePremulIrradiance = PassUav("gPremulIrradiance", LIGHT_TREE_REG_PREMUL_IRRADIANCE);
constexpr BindingSlot kTreeCounters         = PassUav("gVoxCounters", LIGHT_TREE_REG_COUNTERS);
constexpr BindingSlot kTreeNodeVisited      = PassUav("gNodeVisited", LIGHT_TREE_REG_NODE_VISITED);
constexpr BindingSlot kTreeTopLevelConstants =
    PassRootConstants("TopLevelTreeCB", LIGHT_TREE_REG_TOP_LEVEL_CB, 4); // mapX, mapY, importanceMode, pad
constexpr BindingSlot kTreeAvgVisibility  = PassUav("gAvgVisibility", LIGHT_TREE_REG_AVG_VISIBILITY);
constexpr BindingSlot kTreeImportanceHeap = PassUav("gSpixelClusterImportanceHeap", LIGHT_TREE_REG_IMPORTANCE_HEAP);
constexpr BindingSlot kTreeVisibilityMask = PassTableEntry("gClusterVisibilityMask", BindingKind::Uav,
                                                       LIGHT_TREE_REG_VISIBILITY_MASK,
                                                       GlobalDescriptor::ClusterVisibilityMask);
constexpr BindingSlot kTreeIndirectArgs = PassUav("gIndirectDispatchArgs", LIGHT_TREE_REG_INDIRECT_ARGS);

constexpr BindingSlot kLightTreeSlots[] = {
    kTreeGridConstants,   kTreeSortKeys,        kTreeNodes,          kTreeLeafRanges,     kTreeCompactToLeaf,
    kTreeClusterRoots,    kTreeDispatchArgs,    kTreeCompactIds,     kTreeAssignments,    kTreePremulIrradiance,
    kTreeCounters,        kTreeNodeVisited,     kTreeTopLevelConstants, kTreeAvgVisibility, kTreeImportanceHeap,
    kTreeVisibilityMask,  kTreeIndirectArgs};

constexpr uint64_t IndirectArgOffset(uint32_t slot)
{
    return static_cast<uint64_t>(slot) * sizeof(D3D12_DISPATCH_ARGUMENTS);
}
} // namespace

namespace
{
    // uint16 node-index ceiling (2N-1 must fit uint16 => N <= 32768).
    constexpr uint32_t kMaxLeaves = Constants::Graphics::LIGHT_TREE_MAX_LEAVES;
    // Node array holds 2N-1 entries.
    constexpr uint32_t kNodeCapacity = 2 * kMaxLeaves - 1; // 65535
    constexpr uint32_t kCompactCapacity = Constants::Graphics::VOXEL_GUIDING_CAPACITY;
    // Byte offset of numValidVoxels inside TreeBuildDispatchArgs (int3 first).
    constexpr uint32_t kCounterByteOffset = 12;
    constexpr uint32_t kSuperpixelSize = Constants::Graphics::SUPERPIXEL_SIZE;
    constexpr uint32_t kClusterCount = 32;
}

void VxpgLightTreePass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgClusterPass>       clusterPass,
    std::shared_ptr<VxpgClusterVisibilityPass> clusterVisibilityPass)
{
    spdlog::info("Initializing VXPG light tree pass...");

    m_device      = device;
    m_commandList = commandList;
    m_voxelPass   = std::move(voxelPass);
    m_buildPass   = std::move(buildPass);
    m_clusterPass = std::move(clusterPass);
    m_clusterVisibilityPass = std::move(clusterVisibilityPass);

    m_sort.Initialize(device, commandList);

    CreateBuffers();
    CreateRootSignature();
    CreatePSOs();
    CreateCommandSignature();

    m_initialized = true;
}

void VxpgLightTreePass::CreateBuffers()
{
    m_sortKeys = std::make_unique<RWStructuredBuffer<uint64_t>>(
        m_device, BitonicSortPass::kCapacity, L"LightTree SortKeys");
    m_nodes = std::make_unique<RWStructuredBuffer<LightTreeNodeGpu>>(
        m_device, kNodeCapacity, L"LightTree Nodes");
    m_leafRanges = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, kNodeCapacity, L"LightTree LeafRanges");
    m_compactToLeaf = std::make_unique<RWStructuredBuffer<int32_t>>(
        m_device, kCompactCapacity, L"LightTree CompactToLeaf");
    m_clusterRoots = std::make_unique<RWStructuredBuffer<int32_t>>(
        m_device, 32, L"LightTree ClusterRoots");
    m_dispatchArgs = std::make_unique<RWStructuredBuffer<TreeBuildDispatchArgsGpu>>(
        m_device, 1, L"LightTree DispatchArgs");
    m_nodeVisited = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, kNodeCapacity, L"LightTree NodeVisited");
    m_indirectDispatchArgs = std::make_unique<RWStructuredBuffer<DispatchArgsGpu>>(
        m_device, LIGHT_TREE_INDIRECT_SLOT_COUNT, L"LightTree IndirectDispatchArgs");
}

void VxpgLightTreePass::CreateCommandSignature()
{
    D3D12_INDIRECT_ARGUMENT_DESC arg = {};
    arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC desc = {};
    desc.ByteStride       = sizeof(D3D12_DISPATCH_ARGUMENTS);
    desc.NumArgumentDescs = 1;
    desc.pArgumentDescs   = &arg;

    ThrowIfFailed(m_device->CreateCommandSignature(&desc, nullptr,
        IID_PPV_ARGS(&m_dispatchCommandSignature)));
    m_dispatchCommandSignature->SetName(L"VxpgLightTree DispatchCommandSignature");
}

void VxpgLightTreePass::CreateRootSignature()
{
    m_rootSig = RootSignatureBuilder(L"VxpgLightTree RootSig", /*tableCount*/ 1)
                    .Add(kLightTreeSlots)
                    .Build(m_device.Get());
}

void VxpgLightTreePass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgLightTree.clearleaf.shader", L"LightTree Clear PSO");
    m_encodeProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgLightTree.encode.shader", L"LightTree Encode PSO");
    m_initialProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgLightTree.initial.shader", L"LightTree Initial PSO");
    m_internalProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgLightTree.internal.shader", L"LightTree Internal PSO");
    m_mergeProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgLightTree.merge.shader", L"LightTree Merge PSO");
    m_topLevelProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgLightTree.toplevel.shader", L"LightTree TopLevel PSO");
}

void VxpgLightTreePass::OnResize(uint32_t width, uint32_t height)
{
    m_mapX = (width  + kSuperpixelSize - 1) / kSuperpixelSize;
    m_mapY = (height + kSuperpixelSize - 1) / kSuperpixelSize;
    // Implicit 64-slot binary heap per superpixel (SIByL tltree).
    m_spixelClusterHeap = std::make_unique<RWStructuredBuffer<float>>(
        m_device, std::max(1u, m_mapX * m_mapY * 64u), L"LightTree SpixelClusterHeap");
}

// Binds the shared heap, the tree root signature and every root resource. Called
// by each kernel node: the bitonic sort swaps in its own root signature, and
// separate nodes may have barriers placed between them, so none of them may
// inherit another's root state.
bool VxpgLightTreePass::BindRoots()
{
    if (!m_initialized || !m_voxelPass || !m_buildPass || !m_clusterPass)
        return false;

    auto* cmd = m_commandList.Get();

    // The top-level tree's mask descriptor table needs the shared heap bound.
    ID3D12DescriptorHeap* heaps[] = { GlobalDescriptorHeap::Get().GetHeap() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    cmd->SetComputeRootSignature(m_rootSig.Get());
    m_rootSig.Set(cmd, kTreeGridConstants, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeSortKeys, m_sortKeys->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeNodes, m_nodes->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeLeafRanges, m_leafRanges->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeCompactToLeaf, m_compactToLeaf->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeClusterRoots, m_clusterRoots->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeDispatchArgs, m_dispatchArgs->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeCompactIds, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeAssignments, m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreePremulIrradiance, m_buildPass->GetPremulIrradianceBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeCounters, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeNodeVisited, m_nodeVisited->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeIndirectArgs, m_indirectDispatchArgs->GetGPUVirtualAddress());

    return true;
}

// Reset compact->leaf to -1 and NULL-pad the whole sort-key buffer. This one stays
// a fixed worst-case dispatch on purpose: it is what guarantees every key past the
// live count is padding, which is the precondition the shrunk sort ladder relies on
// (the outer stage reads its partner index unguarded). Sizing this to the live
// count too would make the two depend on each other's rounding.
void VxpgLightTreePass::RunClear()
{
    if (!BindRoots())
        return;

    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());
    CommandContext::Get().Dispatch((kCompactCapacity + 255) / 256, 1, 1);
}

// Encode leaf sort keys + dispatch args (+ overflow flag). Fixed dispatch: this is
// the kernel that COMPUTES the indirect args, so it cannot be sized by them, and
// the only pre-existing count (the unclamped lit-voxel total) would over-dispatch
// it by 4x on an overflow frame rather than under-dispatch.
void VxpgLightTreePass::RunEncode()
{
    if (!BindRoots())
        return;

    m_commandList->SetPipelineState(m_encodeProgram->GetPipelineState());
    CommandContext::Get().Dispatch((kMaxLeaves + 255) / 256, 1, 1);
}

// Sort the keys so each cluster is a contiguous Morton run. Swaps in the sort's
// own root signature, which is why every node re-binds.
void VxpgLightTreePass::RunSort()
{
    if (!BindRoots())
        return;

    m_sort.Sort(m_sortKeys->GetUnderlyingResource().Get(),
                m_sortKeys->GetGPUVirtualAddress(),
                m_dispatchArgs->GetGPUVirtualAddress(),
                kCounterByteOffset,
                m_dispatchCommandSignature.Get(),
                m_indirectDispatchArgs->GetUnderlyingResource().Get(),
                IndirectArgOffset(LIGHT_TREE_INDIRECT_SLOT_SORT));
}

// Initialize the 2N-1 node array (leaves get AABB / intensity / cluster).
void VxpgLightTreePass::RunInitial()
{
    if (!BindRoots())
        return;

    m_commandList->SetPipelineState(m_initialProgram->GetPipelineState());
    CommandContext::Get().DispatchIndirect(m_dispatchCommandSignature.Get(),
        m_indirectDispatchArgs->GetUnderlyingResource().Get(), IndirectArgOffset(LIGHT_TREE_INDIRECT_SLOT_NODE));
}

// Build the Karras hierarchy (child + parent links).
void VxpgLightTreePass::RunInternal()
{
    if (!BindRoots())
        return;

    m_commandList->SetPipelineState(m_internalProgram->GetPipelineState());
    CommandContext::Get().DispatchIndirect(m_dispatchCommandSignature.Get(),
        m_indirectDispatchArgs->GetUnderlyingResource().Get(), IndirectArgOffset(LIGHT_TREE_INDIRECT_SLOT_INTERNAL));
}

// Merge bottom-up: AABB + intensity + per-cluster root detection.
void VxpgLightTreePass::RunMerge()
{
    if (!BindRoots())
        return;

    m_commandList->SetPipelineState(m_mergeProgram->GetPipelineState());
    CommandContext::Get().DispatchIndirect(m_dispatchCommandSignature.Get(),
        m_indirectDispatchArgs->GetUnderlyingResource().Get(), IndirectArgOffset(LIGHT_TREE_INDIRECT_SLOT_MERGE));
}

// Top-level tree: per-superpixel implicit heap of the 32 clusters' view-weighted
// importance. Consumes the just-built cluster roots + node intensities and the
// cluster-visibility pass's mask / avg buffers.
void VxpgLightTreePass::RunTopLevel()
{
    auto* avgVisibility = m_clusterVisibilityPass ? m_clusterVisibilityPass->GetAvgVisibilityBuffer() : nullptr;
    if (!m_spixelClusterHeap || !avgVisibility || m_mapX == 0 || !BindRoots())
        return;

    auto* cmd = m_commandList.Get();

    // The avg-visibility buffer stays bound in every mode: IntensityOnly leaves it
    // unwritten (its producer is culled) but the root signature still needs a VA.
    const uint32_t constants[4] = {
        m_mapX, m_mapY,
        static_cast<uint32_t>(g_topLevelImportance.Get()),
        0u
    };
    m_rootSig.SetConstants(cmd, kTreeTopLevelConstants, constants, 4);
    m_rootSig.Set(cmd, kTreeAvgVisibility, avgVisibility->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kTreeImportanceHeap, m_spixelClusterHeap->GetGPUVirtualAddress());
    m_rootSig.SetTable(cmd, 0, GlobalDescriptorHeap::Get().GpuStart());

    // One warp (32 lanes) per superpixel; 8 warps per group => ceil(mapX/8)
    // groups wide, mapY tall (SIByL dispatch (5,23) for its 40x23 map).
    cmd->SetPipelineState(m_topLevelProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_mapX + 7) / 8, m_mapY, 1);
}

// P5. The node array is what LIGHT_TREE_MAX_LEAVES caps, so this row is where the uint16
// node index shows up as a number rather than as a note.
void VxpgLightTreePass::ReportMemory(GpuMemoryReport& report) const
{
    using namespace GpuMemoryStage;
    report.Add(LightTree, "sort keys",              m_sortKeys.get());
    report.Add(LightTree, "tree nodes",             m_nodes.get());
    report.Add(LightTree, "leaf ranges",            m_leafRanges.get());
    report.Add(LightTree, "compact to leaf",        m_compactToLeaf.get());
    report.Add(LightTree, "cluster roots",          m_clusterRoots.get());
    report.Add(LightTree, "tree dispatch args",     m_dispatchArgs.get());
    report.Add(LightTree, "indirect dispatch args", m_indirectDispatchArgs.get());
    report.Add(LightTree, "node visited",           m_nodeVisited.get());
    report.Add(LightTree, "superpixel cluster heap", m_spixelClusterHeap.get());
}
