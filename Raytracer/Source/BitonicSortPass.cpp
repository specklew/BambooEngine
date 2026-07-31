#include "pch.h"
#include "BitonicSortPass.h"

#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

void BitonicSortPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    spdlog::info("Initializing bitonic sort pass...");
    m_device      = device;
    m_commandList = commandList;

    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void BitonicSortPass::CreateRootSignature()
{
    // b0 root constants (k, j, counterOffset); u0 key buffer; u1 counter buffer.
    CD3DX12_ROOT_PARAMETER params[3];
    params[0].InitAsConstants(3, 0);
    params[1].InitAsUnorderedAccessView(0);
    params[2].InitAsUnorderedAccessView(1);

    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)));
    m_rootSig->SetName(L"BitonicSort RootSig");
}

void BitonicSortPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_presortProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgBitonicSort.presort.shader", L"BitonicSort Presort PSO");
    m_outerProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgBitonicSort.outer.shader", L"BitonicSort Outer PSO");
    m_innerProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgBitonicSort.inner.shader", L"BitonicSort Inner PSO");
}

void BitonicSortPass::Sort(
    ID3D12Resource*           keyBuffer,
    D3D12_GPU_VIRTUAL_ADDRESS keyBufferVA,
    D3D12_GPU_VIRTUAL_ADDRESS counterBufferVA,
    uint32_t                  counterByteOffset)
{
    if (!m_initialized)
        return;

    auto* cmd = m_commandList.Get();

    cmd->SetComputeRootSignature(m_rootSig.Get());
    cmd->SetComputeRootUnorderedAccessView(1, keyBufferVA);
    cmd->SetComputeRootUnorderedAccessView(2, counterBufferVA);

    // 65536 elements: presort/inner sort 2048 per group, outer compares 1024
    // pairs per group -> every stage is exactly 32 groups worst-case.
    constexpr uint32_t kGroups = kCapacity / 2048; // 32

    auto keyBarrier = [&]()
    {
        auto b = CD3DX12_RESOURCE_BARRIER::UAV(keyBuffer);
        cmd->ResourceBarrier(1, &b);
    };
    auto setConstants = [&](uint32_t k, uint32_t j)
    {
        uint32_t c[3] = { k, j, counterByteOffset };
        cmd->SetComputeRoot32BitConstants(0, 3, c, 0);
    };

    // Presort: sort each 2048-block into bitonic order.
    cmd->SetPipelineState(m_presortProgram->GetPipelineState());
    setConstants(0, 0);
    cmd->Dispatch(kGroups, 1, 1);
    keyBarrier();

    // Outer/inner ladder: 1 presort + 15 outer + 5 inner = 21 dispatches.
    for (uint32_t k = 4096; k <= kCapacity; k *= 2)
    {
        for (uint32_t j = k / 2; j >= 2048; j /= 2)
        {
            cmd->SetPipelineState(m_outerProgram->GetPipelineState());
            setConstants(k, j);
            cmd->Dispatch(kGroups, 1, 1);
            keyBarrier();
        }
        cmd->SetPipelineState(m_innerProgram->GetPipelineState());
        setConstants(k, 0);
        cmd->Dispatch(kGroups, 1, 1);
        keyBarrier();
    }
}
