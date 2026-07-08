#include "pch.h"
#include "VxpgLightTreePass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgClusterPass.h"
#include "VxpgClusterVisibilityPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/CVars.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

// SIByL ships the top-level tree with visibility = 1 (Average / soft weighting).
// 0 switches to Binary (a cluster is fully in or fully out per superpixel). Both
// signals are produced by the cluster-visibility pass every frame, so this is a
// quality knob, not a cost knob.
static AutoCVarInt g_topLevelUseAvgVisibility("vxpg.topLevelTree.useAvgVisibility",
    "Weight top-level cluster importance by soft avg-visibility (1=SIByL Average, 0=Binary mask)",
    1, CVarFlags::EditCheckbox);

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
    ComPtr<ID3D12DescriptorHeap>       globalHeap,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgClusterPass>       clusterPass,
    std::shared_ptr<VxpgClusterVisibilityPass> clusterVisibilityPass)
{
    spdlog::info("Initializing VXPG light tree pass...");

    m_device      = device;
    m_commandList = commandList;
    m_globalHeap  = globalHeap;
    m_voxelPass   = std::move(voxelPass);
    m_buildPass   = std::move(buildPass);
    m_clusterPass = std::move(clusterPass);
    m_clusterVisibilityPass = std::move(clusterVisibilityPass);

    m_sort.Initialize(device, commandList);

    CreateBuffers();
    CreateRootSignature();
    CreatePSOs();

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
}

void VxpgLightTreePass::CreateRootSignature()
{
    // b0 grid CBV + u0..u10 root UAVs (keys, nodes, leaf ranges, compact->leaf,
    // cluster roots, args, compact ids, cluster assignments, premul irradiance,
    // lit-voxel counter, merge visited-gate). Top-level tree adds b1 root
    // constants (map dims + mode), u11 avg-visibility, u12 heap, and the mask
    // texture as a shared-heap descriptor table (u13, at slot 530).
    CD3DX12_DESCRIPTOR_RANGE maskRange;
    maskRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 13, 0,
        Constants::Graphics::CLUSTER_VISIBILITY_MASK_DESCRIPTOR_INDEX); // u13 @ 530

    CD3DX12_ROOT_PARAMETER params[16];
    params[0].InitAsConstantBufferView(0);
    for (uint32_t i = 0; i < 11; ++i)
        params[1 + i].InitAsUnorderedAccessView(i);
    params[12].InitAsConstants(4, 1);           // b1: mapX, mapY, useAvgVisibility, pad
    params[13].InitAsUnorderedAccessView(11);   // u11 avg-visibility
    params[14].InitAsUnorderedAccessView(12);   // u12 importance heap
    params[15].InitAsDescriptorTable(1, &maskRange); // u13 visibility mask

    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)));
    m_rootSig->SetName(L"VxpgLightTree RootSig");
}

void VxpgLightTreePass::CreatePSOs()
{
    auto& rm = ResourceManager::Get();

    auto createCsPso = [&](const char* assetPath, const wchar_t* name,
                           ComPtr<ID3D12PipelineState>& out)
    {
        auto handle = rm.GetOrLoadShader(AssetId(assetPath));
        auto blob = rm.shaders.GetResource(handle).bytecode;

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_rootSig.Get();
        desc.CS = CD3DX12_SHADER_BYTECODE(blob->GetBufferPointer(), blob->GetBufferSize());
        ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out)));
        out->SetName(name);
    };

    createCsPso("resources/shaders/vxpgLightTree.clearleaf.shader", L"LightTree Clear PSO",    m_clearPso);
    createCsPso("resources/shaders/vxpgLightTree.encode.shader",    L"LightTree Encode PSO",   m_encodePso);
    createCsPso("resources/shaders/vxpgLightTree.initial.shader",   L"LightTree Initial PSO",  m_initialPso);
    createCsPso("resources/shaders/vxpgLightTree.internal.shader",  L"LightTree Internal PSO", m_internalPso);
    createCsPso("resources/shaders/vxpgLightTree.merge.shader",     L"LightTree Merge PSO",    m_mergePso);
    createCsPso("resources/shaders/vxpgLightTree.toplevel.shader",  L"LightTree TopLevel PSO", m_topLevelPso);
}

void VxpgLightTreePass::OnResize(uint32_t width, uint32_t height)
{
    m_mapX = (width  + kSuperpixelSize - 1) / kSuperpixelSize;
    m_mapY = (height + kSuperpixelSize - 1) / kSuperpixelSize;
    // Implicit 64-slot binary heap per superpixel (SIByL tltree).
    m_spixelClusterHeap = std::make_unique<RWStructuredBuffer<float>>(
        m_device, std::max(1u, m_mapX * m_mapY * 64u), L"LightTree SpixelClusterHeap");
}

void VxpgLightTreePass::Run()
{
    if (!m_initialized || !m_voxelPass || !m_buildPass || !m_clusterPass)
        return;

    auto* cmd = m_commandList.Get();

    // Bind the shared heap up front: it stays bound through the bitonic sort
    // (which uses only root descriptors) and is needed by the top-level tree's
    // mask descriptor table at the tail.
    ID3D12DescriptorHeap* heaps[] = { m_globalHeap.Get() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    // Binds the tree root sig + all resources (re-called after the sort swaps in
    // its own root signature).
    auto bindRoots = [&]()
    {
        cmd->SetComputeRootSignature(m_rootSig.Get());
        cmd->SetComputeRootConstantBufferView(0, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(1, m_sortKeys->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(2, m_nodes->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(3, m_leafRanges->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(4, m_compactToLeaf->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(5, m_clusterRoots->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(6, m_dispatchArgs->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(7, m_buildPass->GetCompactIdsBuffer()->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(8, m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(9, m_buildPass->GetPremulIrradianceBuffer()->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(10, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(11, m_nodeVisited->GetGPUVirtualAddress());
    };

    bindRoots();

    // Fixed worst-case dispatches sized to the uint16 leaf cap (ExecuteIndirect
    // deferred, ADR 0003 option b); in-shader guards early-out past the count.
    const uint32_t clearGroups    = (kCompactCapacity + 255) / 256; // 512
    const uint32_t leafGroups     = (kMaxLeaves + 255) / 256;       // 128
    const uint32_t nodeGroups     = (kNodeCapacity + 255) / 256;    // 256
    const uint32_t internalGroups = (kMaxLeaves - 1 + 255) / 256;   // 128

    // Reset compact->leaf to -1 and NULL-pad the whole sort-key buffer (so the
    // fixed 65536 sort network's over-dispatch reads padding, not stale garbage).
    cmd->SetPipelineState(m_clearPso.Get());
    cmd->Dispatch(clearGroups, 1, 1);
    m_compactToLeaf->UavBarrier(cmd);
    m_sortKeys->UavBarrier(cmd);

    // Encode leaf sort keys + dispatch args (+ overflow flag).
    cmd->SetPipelineState(m_encodePso.Get());
    cmd->Dispatch(leafGroups, 1, 1);
    m_sortKeys->UavBarrier(cmd);
    m_dispatchArgs->UavBarrier(cmd);

    // Sort the keys so each cluster is a contiguous Morton run (swaps root sig).
    m_sort.Sort(m_sortKeys->GetUnderlyingResource().Get(),
                m_sortKeys->GetGPUVirtualAddress(),
                m_dispatchArgs->GetGPUVirtualAddress(),
                kCounterByteOffset);
    m_sortKeys->UavBarrier(cmd);

    bindRoots();

    // Initialize the 2N-1 node array (leaves get AABB / intensity / cluster).
    cmd->SetPipelineState(m_initialPso.Get());
    cmd->Dispatch(nodeGroups, 1, 1);
    m_nodes->UavBarrier(cmd);
    m_compactToLeaf->UavBarrier(cmd);
    m_clusterRoots->UavBarrier(cmd);

    // Build the Karras hierarchy (child + parent links).
    cmd->SetPipelineState(m_internalPso.Get());
    cmd->Dispatch(internalGroups, 1, 1);
    m_nodes->UavBarrier(cmd);

    // Merge bottom-up: AABB + intensity + per-cluster root detection.
    cmd->SetPipelineState(m_mergePso.Get());
    cmd->Dispatch(leafGroups, 1, 1);
    m_nodes->UavBarrier(cmd);
    m_clusterRoots->UavBarrier(cmd);

    // Top-level tree: per-superpixel implicit heap of the 32 clusters' view-
    // weighted importance. Consumes the just-built cluster roots + node
    // intensities and the cluster-visibility pass's mask / avg buffers.
    auto* avgVisibility = m_clusterVisibilityPass ? m_clusterVisibilityPass->GetAvgVisibilityBuffer() : nullptr;
    if (m_spixelClusterHeap && avgVisibility && m_mapX > 0)
    {
        const uint32_t constants[4] = {
            m_mapX, m_mapY,
            static_cast<uint32_t>(g_topLevelUseAvgVisibility.Get() != 0),
            0u
        };
        cmd->SetComputeRoot32BitConstants(12, 4, constants, 0);
        cmd->SetComputeRootUnorderedAccessView(13, avgVisibility->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(14, m_spixelClusterHeap->GetGPUVirtualAddress());
        cmd->SetComputeRootDescriptorTable(15, m_globalHeap->GetGPUDescriptorHandleForHeapStart());

        // One warp (32 lanes) per superpixel; 8 warps per group => ceil(mapX/8)
        // groups wide, mapY tall (SIByL dispatch (5,23) for its 40x23 map).
        cmd->SetPipelineState(m_topLevelPso.Get());
        cmd->Dispatch((m_mapX + 7) / 8, m_mapY, 1);
        m_spixelClusterHeap->UavBarrier(cmd);
    }
}
