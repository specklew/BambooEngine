#include "pch.h"
#include "SupervoxelClusterPass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

void SupervoxelClusterPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>  voxelPass)
{
    spdlog::info("Initializing supervoxel cluster pass...");

    m_device      = device;
    m_commandList = commandList;
    m_voxelPass   = std::move(voxelPass);

    CreateBuffers();
    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void SupervoxelClusterPass::CreateBuffers()
{
    constexpr uint32_t maxSv = Constants::Graphics::MAX_SUPERVOXELS;
    m_svIrradiance = std::make_unique<RWStructuredBuffer<uint32_t>>(m_device, maxSv, L"Supervoxel Irradiance");
    m_svCount      = std::make_unique<RWStructuredBuffer<uint32_t>>(m_device, maxSv, L"Supervoxel Count");
}

void SupervoxelClusterPass::CreateRootSignature()
{
    // Table: u0 = irradiance, u1 = vpl count (slots 1..2 of the voxelization
    // pass private heap). Root UAVs: u2 supervoxel irradiance, u3 supervoxel
    // count. Root constants b0: gridDim, clusterFactor, svDim.
    CD3DX12_DESCRIPTOR_RANGE texRange;
    texRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0);

    CD3DX12_ROOT_PARAMETER params[4];
    params[0].InitAsConstants(4, 0);
    params[1].InitAsDescriptorTable(1, &texRange);
    params[2].InitAsUnorderedAccessView(2);
    params[3].InitAsUnorderedAccessView(3);

    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)));
    m_rootSig->SetName(L"SupervoxelCluster RootSig");
}

void SupervoxelClusterPass::CreatePSOs()
{
    auto& rm = ResourceManager::Get();

    auto createCsPso = [&](const char* assetPath, const wchar_t* name, ComPtr<ID3D12PipelineState>& out)
    {
        auto handle = rm.GetOrLoadShader(AssetId(assetPath));
        auto blob = rm.shaders.GetResource(handle).bytecode;

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_rootSig.Get();
        desc.CS = CD3DX12_SHADER_BYTECODE(blob->GetBufferPointer(), blob->GetBufferSize());
        ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out)));
        out->SetName(name);
    };

    createCsPso("resources/shaders/supervoxelCluster.clear.shader", L"Supervoxel Clear PSO", m_clearPso);
    createCsPso("resources/shaders/supervoxelCluster.accum.shader", L"Supervoxel Accum PSO", m_accumPso);
}

void SupervoxelClusterPass::Run()
{
    if (!m_initialized || !m_voxelPass)
        return;

    auto heap = m_voxelPass->GetDescriptorHeap();
    if (!heap)
        return;

    const uint32_t gridDim       = m_voxelPass->GetGridDim();
    const uint32_t clusterFactor = Constants::Graphics::SUPERVOXEL_GRID_FACTOR;
    m_svDim = (gridDim + clusterFactor - 1) / clusterFactor;

    // GPU handle to slot 1 (irradiance) of the voxelization pass heap; the
    // 2-descriptor table covers irradiance + vpl count.
    UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE texTable = heap->GetGPUDescriptorHandleForHeapStart();
    texTable.ptr += inc;

    ID3D12DescriptorHeap* heaps[] = { heap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSig.Get());

    uint32_t constants[4] = { gridDim, clusterFactor, m_svDim, 0 };
    m_commandList->SetComputeRoot32BitConstants(0, 4, constants, 0);
    m_commandList->SetComputeRootDescriptorTable(1, texTable);
    m_commandList->SetComputeRootUnorderedAccessView(2, m_svIrradiance->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(3, m_svCount->GetGPUVirtualAddress());

    auto* cmd = m_commandList.Get();

    constexpr uint32_t clearGroups = (Constants::Graphics::MAX_SUPERVOXELS + 255) / 256;
    m_commandList->SetPipelineState(m_clearPso.Get());
    m_commandList->Dispatch(clearGroups, 1, 1);
    m_svIrradiance->UavBarrier(cmd);
    m_svCount->UavBarrier(cmd);

    m_commandList->SetPipelineState(m_accumPso.Get());
    const uint32_t groups = (gridDim + 7) / 8;
    m_commandList->Dispatch(groups, groups, groups);
    m_svIrradiance->UavBarrier(cmd);
    m_svCount->UavBarrier(cmd);
}
