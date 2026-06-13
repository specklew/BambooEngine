#include "pch.h"
#include "VoxelGuidingBuildPass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"

using Microsoft::WRL::ComPtr;

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
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    constexpr uint32_t capacity = Constants::Graphics::VOXEL_GUIDING_CAPACITY;

    auto createUavBuffer = [&](uint64_t byteSize, const wchar_t* name, ComPtr<ID3D12Resource>& out)
    {
        // Pad to 256 B: tiny buffers can fail root-UAV alignment requirements
        byteSize = (byteSize + 255ull) & ~255ull;

        auto desc = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        // Buffers are always created in COMMON (InitialState is ignored with a
        // warning); implicit state promotion covers the UAV access.
        HRESULT hr = m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&out));
        if (FAILED(hr))
        {
            spdlog::error("VoxelGuidingBuildPass: buffer creation failed (size={}, hr={:#010x}, deviceRemoved={:#010x})",
                byteSize, static_cast<uint32_t>(hr),
                static_cast<uint32_t>(m_device->GetDeviceRemovedReason()));
            ThrowIfFailed(hr);
        }
        out->SetName(name);
    };

    createUavBuffer(2 * sizeof(uint32_t),        L"VoxelGuiding Counters",   m_counters);
    createUavBuffer(capacity * sizeof(uint32_t), L"VoxelGuiding CompactIds", m_compactIds);
    createUavBuffer(capacity * sizeof(float),    L"VoxelGuiding Weights",    m_weights);
    createUavBuffer(capacity * sizeof(float),    L"VoxelGuiding Cdf",        m_cdf);
}

void VoxelGuidingBuildPass::CreateRootSignature()
{
    // Table: u0 = irradiance, u1 = vpl count (slots 1..2 of the voxelization
    // pass private heap). Root UAVs: u2 counters, u3 compactIds, u4 weights,
    // u5 cdf. Root constants b0: gridDim.
    CD3DX12_DESCRIPTOR_RANGE texRange;
    texRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0);

    CD3DX12_ROOT_PARAMETER params[6];
    params[0].InitAsConstants(4, 0);
    params[1].InitAsDescriptorTable(1, &texRange);
    params[2].InitAsUnorderedAccessView(2);
    params[3].InitAsUnorderedAccessView(3);
    params[4].InitAsUnorderedAccessView(4);
    params[5].InitAsUnorderedAccessView(5);

    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)));
    m_rootSig->SetName(L"VoxelGuidingBuild RootSig");
}

void VoxelGuidingBuildPass::CreatePSOs()
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

    createCsPso("resources/shaders/voxelGuidingBuild.clear.shader",   L"VoxelGuiding Clear PSO",   m_clearPso);
    createCsPso("resources/shaders/voxelGuidingBuild.compact.shader", L"VoxelGuiding Compact PSO", m_compactPso);
    createCsPso("resources/shaders/voxelGuidingBuild.cdf.shader",     L"VoxelGuiding Cdf PSO",     m_cdfPso);
}

void VoxelGuidingBuildPass::Run()
{
    if (!m_initialized || !m_voxelPass)
        return;

    auto heap = m_voxelPass->GetDescriptorHeap();
    if (!heap)
        return;

    const uint32_t gridDim = m_voxelPass->GetGridDim();

    // GPU handle to slot 1 (irradiance) of the voxelization pass heap; the
    // 2-descriptor table covers irradiance + vpl count.
    UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE texTable = heap->GetGPUDescriptorHandleForHeapStart();
    texTable.ptr += inc;

    ID3D12DescriptorHeap* heaps[] = { heap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSig.Get());

    uint32_t constants[4] = { gridDim, 0, 0, 0 };
    m_commandList->SetComputeRoot32BitConstants(0, 4, constants, 0);
    m_commandList->SetComputeRootDescriptorTable(1, texTable);
    m_commandList->SetComputeRootUnorderedAccessView(2, m_counters->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(3, m_compactIds->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(4, m_weights->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(5, m_cdf->GetGPUVirtualAddress());

    auto uavBarrier = [&](ID3D12Resource* res)
    {
        auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(res);
        m_commandList->ResourceBarrier(1, &barrier);
    };

    m_commandList->SetPipelineState(m_clearPso.Get());
    m_commandList->Dispatch(1, 1, 1);
    uavBarrier(m_counters.Get());

    m_commandList->SetPipelineState(m_compactPso.Get());
    const uint32_t groups = (gridDim + 7) / 8;
    m_commandList->Dispatch(groups, groups, groups);
    uavBarrier(m_counters.Get());
    uavBarrier(m_compactIds.Get());
    uavBarrier(m_weights.Get());

    m_commandList->SetPipelineState(m_cdfPso.Get());
    m_commandList->Dispatch(1, 1, 1);
    uavBarrier(m_counters.Get());
    uavBarrier(m_cdf.Get());
}
