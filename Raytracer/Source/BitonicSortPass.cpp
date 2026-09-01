#include "pch.h"
#include "CommandContext.h"
#include "BitonicSortPass.h"

#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
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

static constexpr BindingSlot kSortConstants = PassRootConstants("BitonicCB", BITONIC_REG_CB, 3); // k, j, counterOffset
static constexpr BindingSlot kSortBuffer    = PassUav("gSortBuffer", BITONIC_REG_SORT_BUFFER);
static constexpr BindingSlot kSortCounter   = PassUav("gCounter", BITONIC_REG_COUNTER);

void BitonicSortPass::CreateRootSignature()
{
    m_rootSig = RootSignatureBuilder(L"BitonicSort RootSig", /*tableCount*/ 0)
                    .Add(kSortConstants)
                    .Add(kSortBuffer)
                    .Add(kSortCounter)
                    .Build(m_device.Get());
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
    uint32_t                  counterByteOffset,
    ID3D12CommandSignature*   dispatchSignature,
    ID3D12Resource*           stageArgsBuffer,
    uint64_t                  stageArgsByteOffset)
{
    if (!m_initialized)
        return;

    auto* cmd = m_commandList.Get();

    cmd->SetComputeRootSignature(m_rootSig.Get());
    m_rootSig.Set(cmd, kSortBuffer, keyBufferVA);
    m_rootSig.Set(cmd, kSortCounter, counterBufferVA);

    // Group counts come from the producer, one triple per stage in ladder order.
    // This loop and the one that fills the buffer must stay identical.
    uint32_t stage = 0;
    auto dispatchStage = [&]()
    {
        CommandContext::Get().DispatchIndirect(dispatchSignature, stageArgsBuffer,
            stageArgsByteOffset + static_cast<uint64_t>(stage++) * sizeof(D3D12_DISPATCH_ARGUMENTS));
    };

    auto keyBarrier = [&]()
    {
        CommandContext::Get().UavBarrierRaw(keyBuffer);
    };
    auto setConstants = [&](uint32_t k, uint32_t j)
    {
        uint32_t c[3] = { k, j, counterByteOffset };
        m_rootSig.SetConstants(cmd, kSortConstants, c, 3);
    };

    // Presort: sort each 2048-block into bitonic order.
    cmd->SetPipelineState(m_presortProgram->GetPipelineState());
    setConstants(0, 0);
    dispatchStage();
    keyBarrier();

    // Outer/inner ladder: 1 presort + 21 outer + 6 inner = 28 dispatches.
    for (uint32_t k = 4096; k <= kCapacity; k *= 2)
    {
        for (uint32_t j = k / 2; j >= 2048; j /= 2)
        {
            cmd->SetPipelineState(m_outerProgram->GetPipelineState());
            setConstants(k, j);
            dispatchStage();
            keyBarrier();
        }
        cmd->SetPipelineState(m_innerProgram->GetPipelineState());
        setConstants(k, 0);
        dispatchStage();
        keyBarrier();
    }

    assert(stage == BITONIC_SORT_STAGE_COUNT &&
           "ladder length drifted from the encode kernel's — stages would read the wrong group counts");
}
