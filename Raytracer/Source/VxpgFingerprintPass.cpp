#include "pch.h"
#include "VxpgFingerprintPass.h"

#include "Constants.h"
#include "VoxelGuidingBuildPass.h"
#include "LightInjectionPass.h"
#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
    constexpr uint32_t kRepresentativeCount = 128;      // 16 x 8
    constexpr uint32_t kFingerprintMaskWords = 4;       // 128 bits / 32
    constexpr uint32_t kDispatchArgsEntries = 3;

    // PCG hash — matches Random.hlsl's pcg_hash so the CPU seed decorrelates
    // the per-frame stratified picks the same way the shader RNG expects.
    uint32_t PcgHash(uint32_t state)
    {
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
}

void VxpgFingerprintPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<LightInjectionPass>    injectionPass)
{
    spdlog::info("Initializing VXPG fingerprint pass...");

    m_device        = device;
    m_commandList   = commandList;
    m_buildPass     = std::move(buildPass);
    m_injectionPass = std::move(injectionPass);

    CreateBuffers();
    CreatePrivateHeap();
    CreateRootSignatures();
    CreatePSOs();
    CreateCommandSignature();

    m_initialized = true;
}

void VxpgFingerprintPass::CreateBuffers()
{
    constexpr uint32_t capacity = Constants::Graphics::VOXEL_GUIDING_CAPACITY;

    m_screenRepresentativePoints = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT4>>(
        m_device, kRepresentativeCount, L"Fingerprint ScreenRepresentativePoints");
    m_guidingDispatchArgs = std::make_unique<RWStructuredBuffer<DirectX::XMUINT4>>(
        m_device, kDispatchArgsEntries, L"Fingerprint GuidingDispatchArgs");
    m_voxelFingerprints = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, capacity * kFingerprintMaskWords, L"Fingerprint VoxelFingerprints");
}

void VxpgFingerprintPass::CreatePrivateHeap()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 1; // ShadingPoints UAV
    heapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_descHeap)));
    m_descHeap->SetName(L"VxpgFingerprint Heap");
}

void VxpgFingerprintPass::RebindShadingPointsIfChanged()
{
    ID3D12Resource* shadingPoints =
        m_injectionPass ? m_injectionPass->GetShadingPointsTexture().Get() : nullptr;
    if (!shadingPoints || shadingPoints == m_boundShadingPoints)
        return;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format        = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(shadingPoints, nullptr, &uav,
        m_descHeap->GetCPUDescriptorHandleForHeapStart());
    m_boundShadingPoints = shadingPoints;
}

void VxpgFingerprintPass::CreateRootSignatures()
{
    // Presample: b0 root constants (resolution + seed), table u0 ShadingPoints,
    // root UAVs u1 representatives (out), u2 dispatch args (out), u3 counters (in).
    {
        CD3DX12_DESCRIPTOR_RANGE shadingPointsRange;
        shadingPointsRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0); // u0

        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsConstants(4, 0);
        params[1].InitAsDescriptorTable(1, &shadingPointsRange);
        params[2].InitAsUnorderedAccessView(1);
        params[3].InitAsUnorderedAccessView(2);
        params[4].InitAsUnorderedAccessView(3);

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_NONE);
        ComPtr<ID3DBlob> sig, err;
        ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
        ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
            IID_PPV_ARGS(&m_presampleRootSig)));
        m_presampleRootSig->SetName(L"VxpgFingerprint Presample RootSig");
    }

    // Visibility: t0 TLAS, root UAVs u1 representatives (in), u2 compact light
    // points (in), u3 dispatch args (in, count), u4 fingerprints (out).
    {
        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsShaderResourceView(0); // t0 TLAS
        params[1].InitAsUnorderedAccessView(1);
        params[2].InitAsUnorderedAccessView(2);
        params[3].InitAsUnorderedAccessView(3);
        params[4].InitAsUnorderedAccessView(4);

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_NONE);
        ComPtr<ID3DBlob> sig, err;
        ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
        ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
            IID_PPV_ARGS(&m_visibilityRootSig)));
        m_visibilityRootSig->SetName(L"VxpgFingerprint Visibility RootSig");
    }
}

void VxpgFingerprintPass::CreatePSOs()
{
    auto& rm = ResourceManager::Get();

    auto createCsPso = [&](const char* assetPath, ID3D12RootSignature* rootSig,
                           const wchar_t* name, ComPtr<ID3D12PipelineState>& out)
    {
        auto handle = rm.GetOrLoadShader(AssetId(assetPath));
        auto blob = rm.shaders.GetResource(handle).bytecode;

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSig;
        desc.CS = CD3DX12_SHADER_BYTECODE(blob->GetBufferPointer(), blob->GetBufferSize());
        ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out)));
        out->SetName(name);
    };

    createCsPso("resources/shaders/vxpgFingerprint.presample.shader",  m_presampleRootSig.Get(),
                L"VxpgFingerprint Presample PSO",  m_presamplePso);
    createCsPso("resources/shaders/vxpgFingerprint.visibility.shader", m_visibilityRootSig.Get(),
                L"VxpgFingerprint Visibility PSO", m_visibilityPso);
}

void VxpgFingerprintPass::CreateCommandSignature()
{
    D3D12_INDIRECT_ARGUMENT_DESC arg = {};
    arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC desc = {};
    desc.ByteStride       = sizeof(DirectX::XMUINT4); // one args entry (xyz + count)
    desc.NumArgumentDescs = 1;
    desc.pArgumentDescs   = &arg;

    // Pure dispatch signature: no root arguments change per command, so the root
    // signature is null and this object is reusable for any indirect dispatch.
    ThrowIfFailed(m_device->CreateCommandSignature(&desc, nullptr,
        IID_PPV_ARGS(&m_dispatchCommandSignature)));
    m_dispatchCommandSignature->SetName(L"VxpgFingerprint DispatchCommandSignature");
}

void VxpgFingerprintPass::OnResize(uint32_t width, uint32_t height)
{
    m_width  = width;
    m_height = height;
}

void VxpgFingerprintPass::Run(D3D12_GPU_VIRTUAL_ADDRESS tlasVa, uint32_t frameIndex)
{
    if (!m_initialized || !m_buildPass || !m_injectionPass || tlasVa == 0 ||
        m_width == 0 || m_height == 0)
        return;

    RebindShadingPointsIfChanged();
    if (!m_boundShadingPoints)
        return;

    auto* cmd = m_commandList.Get();

    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    // ---- Kernel 1: pick 128 screen representatives + emit dispatch args ----
    cmd->SetComputeRootSignature(m_presampleRootSig.Get());
    uint32_t presampleConstants[4] = { m_width, m_height,
        PcgHash(frameIndex), 0u };
    cmd->SetComputeRoot32BitConstants(0, 4, presampleConstants, 0);
    cmd->SetComputeRootDescriptorTable(1, m_descHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->SetComputeRootUnorderedAccessView(2, m_screenRepresentativePoints->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(3, m_guidingDispatchArgs->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(4, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());

    cmd->SetPipelineState(m_presamplePso.Get());
    cmd->Dispatch(1, 1, 1); // one 16x8 group = 128 representatives
    m_screenRepresentativePoints->UavBarrier(cmd);
    m_guidingDispatchArgs->UavBarrier(cmd);

    // ---- Kernel 2: fingerprint every lit voxel via inline shadow rays ----
    // Dispatched indirectly off gGuidingDispatchArgs[2] = (4, ceil(litVoxelCount/8),
    // 1): X = 4 groups of 32 = 128 representatives; Y sized to the live lit-voxel
    // count. Replaces the worst-case (4, ceil(CAPACITY/8)=16384, 1) fixed dispatch
    // (ADR 0003 option b). The presample kernel emitted the count this frame.
    cmd->SetComputeRootSignature(m_visibilityRootSig.Get());
    cmd->SetComputeRootShaderResourceView(0, tlasVa);
    cmd->SetComputeRootUnorderedAccessView(1, m_screenRepresentativePoints->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(2, m_buildPass->GetCompactVoxelLightPointsBuffer()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(3, m_guidingDispatchArgs->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(4, m_voxelFingerprints->GetGPUVirtualAddress());
    cmd->SetPipelineState(m_visibilityPso.Get());

    // The args buffer was just written as a UAV; flip it to INDIRECT_ARGUMENT for
    // the ExecuteIndirect read, then back to UAV so the downstream cluster pass
    // and next frame's presample see it as a UAV again.
    ID3D12Resource* argsResource = m_guidingDispatchArgs->GetUnderlyingResource().Get();
    auto toIndirect = CD3DX12_RESOURCE_BARRIER::Transition(argsResource,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    cmd->ResourceBarrier(1, &toIndirect);

    constexpr uint32_t kVisibilityArgsOffset = 2 * sizeof(DirectX::XMUINT4); // entry [2]
    cmd->ExecuteIndirect(m_dispatchCommandSignature.Get(), 1, argsResource,
        kVisibilityArgsOffset, nullptr, 0);

    auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(argsResource,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &toUav);

    m_voxelFingerprints->UavBarrier(cmd);
}
