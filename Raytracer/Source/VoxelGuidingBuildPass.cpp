#include "pch.h"
#include "VoxelGuidingBuildPass.h"

#include "Constants.h"
#include "VoxelizationPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

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
    // Table: u0 irradiance, u1 vpl count, u2 representative VPL (private heap).
    // Root UAVs: u3 counters, u4 compactIds, u5 inverse index, u6/u7 live
    // bounds, u8 representVPL, u9 premul irradiance, u10/u11 baked bounds.
    // Root constants b0: gridDim.
    CD3DX12_DESCRIPTOR_RANGE texRange;
    texRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0);

    CD3DX12_ROOT_PARAMETER params[11];
    params[0].InitAsConstants(4, 0);
    params[1].InitAsDescriptorTable(1, &texRange);
    for (uint32_t i = 0; i < 9; ++i)
        params[2 + i].InitAsUnorderedAccessView(3 + i);

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
    createCsPso("resources/shaders/voxelGuidingBuild.reload.shader",  L"VoxelGuiding Reload PSO",  m_reloadPso);
    createCsPso("resources/shaders/voxelGuidingBuild.compact.shader", L"VoxelGuiding Compact PSO", m_compactPso);
}

void VoxelGuidingBuildPass::Run(ID3D12Resource* representativeTex)
{
    if (!m_initialized || !m_voxelPass)
        return;

    RebindDescriptorsIfChanged(representativeTex);

    const uint32_t gridDim = m_voxelPass->GetGridDim();

    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSig.Get());

    uint32_t constants[4] = { gridDim, 0, 0, 0 };
    m_commandList->SetComputeRoot32BitConstants(0, 4, constants, 0);
    m_commandList->SetComputeRootDescriptorTable(1, m_descHeap->GetGPUDescriptorHandleForHeapStart());
    m_commandList->SetComputeRootUnorderedAccessView(2,  m_counters->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(3,  m_compactIds->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(4,  m_inverseIndex->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(5,  m_liveBoundMin->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(6,  m_liveBoundMax->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(7,  m_compactVoxelLightPoints->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(8,  m_premulIrradiance->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(9,  m_voxelPass->GetBakedBoundMinBuffer()->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(10, m_voxelPass->GetBakedBoundMaxBuffer()->GetGPUVirtualAddress());

    auto* cmd = m_commandList.Get();
    const uint32_t groups = (gridDim + 7) / 8;

    m_commandList->SetPipelineState(m_clearPso.Get());
    m_commandList->Dispatch(1, 1, 1);
    m_counters->UavBarrier(cmd);

    // Reload baked bounds for lit voxels before compaction reads them.
    m_commandList->SetPipelineState(m_reloadPso.Get());
    m_commandList->Dispatch(groups, groups, groups);
    m_liveBoundMin->UavBarrier(cmd);
    m_liveBoundMax->UavBarrier(cmd);

    m_commandList->SetPipelineState(m_compactPso.Get());
    m_commandList->Dispatch(groups, groups, groups);
    m_counters->UavBarrier(cmd);
    m_compactIds->UavBarrier(cmd);
    m_inverseIndex->UavBarrier(cmd);
    m_compactVoxelLightPoints->UavBarrier(cmd);
    m_premulIrradiance->UavBarrier(cmd);
}
