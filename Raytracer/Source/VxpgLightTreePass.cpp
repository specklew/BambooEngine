#include "pch.h"
#include "VxpgLightTreePass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgClusterPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
    // uint16 node-index ceiling (2N-1 must fit uint16 => N <= 32768).
    constexpr uint32_t kMaxLeaves = Constants::Graphics::LIGHT_TREE_MAX_LEAVES;
    // Node array holds 2N-1 entries.
    constexpr uint32_t kNodeCapacity = 2 * kMaxLeaves - 1; // 65535
    constexpr uint32_t kCompactCapacity = Constants::Graphics::VOXEL_GUIDING_CAPACITY;
    // Byte offset of numValidVoxels inside TreeBuildDispatchArgs (int3 first).
    constexpr uint32_t kCounterByteOffset = 12;
}

void VxpgLightTreePass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgClusterPass>       clusterPass)
{
    spdlog::info("Initializing VXPG light tree pass...");

    m_device      = device;
    m_commandList = commandList;
    m_voxelPass   = std::move(voxelPass);
    m_buildPass   = std::move(buildPass);
    m_clusterPass = std::move(clusterPass);

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
    // b0 grid CBV + u0..u9 root UAVs (keys, nodes, leaf ranges, compact->leaf,
    // cluster roots, args, compact ids, cluster assignments, premul irradiance,
    // lit-voxel counter, merge visited-gate).
    CD3DX12_ROOT_PARAMETER params[12];
    params[0].InitAsConstantBufferView(0);
    for (uint32_t i = 0; i < 11; ++i)
        params[1 + i].InitAsUnorderedAccessView(i);

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
}

void VxpgLightTreePass::Run()
{
    if (!m_initialized || !m_voxelPass || !m_buildPass || !m_clusterPass)
        return;

    auto* cmd = m_commandList.Get();

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
}
