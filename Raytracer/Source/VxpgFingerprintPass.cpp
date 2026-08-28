#include "pch.h"
#include "CommandContext.h"
#include "VxpgFingerprintPass.h"

#include "Constants.h"
#include "VoxelGuidingBuildPass.h"
#include "LightInjectionPass.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
    constexpr uint32_t kRepresentativeCount = 128;      // 16 x 8
    constexpr uint32_t kFingerprintMaskWords = 4;       // 128 bits / 32
    constexpr uint32_t kDispatchArgsEntries = 3;

    // Two kernels, two signatures. The visibility kernel deliberately rebinds
    // u1-u3 to different buffers than the presample kernel writes them from, so
    // the two slot tables share register numbers but nothing else.
    constexpr BindingSlot kPresampleConstants = PassRootConstants("PresampleCB", FINGERPRINT_PRESAMPLE_REG_CB, 4);
    constexpr BindingSlot kPresampleShadingPoints =
        PassTableEntry("gShadingPoints", BindingKind::Uav, FINGERPRINT_PRESAMPLE_REG_SHADING_POINTS,
                       GlobalDescriptor::ShadingPoints);
    constexpr BindingSlot kPresampleRepresentatives =
        PassUav("gScreenRepresentativePoints", FINGERPRINT_PRESAMPLE_REG_REPRESENTATIVES);
    constexpr BindingSlot kPresampleDispatchArgs = PassUav("gGuidingDispatchArgs", FINGERPRINT_PRESAMPLE_REG_DISPATCH_ARGS);
    constexpr BindingSlot kPresampleCounters     = PassUav("gGuidingCounters", FINGERPRINT_PRESAMPLE_REG_COUNTERS);

    constexpr BindingSlot kPresampleSlots[] = {kPresampleConstants, kPresampleShadingPoints, kPresampleRepresentatives,
                                               kPresampleDispatchArgs, kPresampleCounters};

    constexpr BindingSlot kVisibilityTlas = PassSrv("gSceneBVH", FINGERPRINT_VISIBILITY_REG_TLAS);
    constexpr BindingSlot kVisibilityRepresentatives =
        PassUav("gReadRepresentativePoints", FINGERPRINT_VISIBILITY_REG_REPRESENTATIVES);
    constexpr BindingSlot kVisibilityLightPoints =
        PassUav("gCompactVoxelLightPoints", FINGERPRINT_VISIBILITY_REG_LIGHT_POINTS);
    constexpr BindingSlot kVisibilityDispatchArgs = PassUav("gReadDispatchArgs", FINGERPRINT_VISIBILITY_REG_DISPATCH_ARGS);
    constexpr BindingSlot kVisibilityFingerprints = PassUav("gVoxelFingerprints", FINGERPRINT_VISIBILITY_REG_FINGERPRINTS);

    constexpr BindingSlot kVisibilityConstants =
        PassRootConstants("VisibilityCB", FINGERPRINT_VISIBILITY_REG_CB, 4);

    constexpr BindingSlot kVisibilitySlots[] = {kVisibilityTlas, kVisibilityRepresentatives, kVisibilityLightPoints,
                                                kVisibilityDispatchArgs, kVisibilityFingerprints,
                                                kVisibilityConstants};

    // Diagnostic decomposition of the visibility test, read out through
    // vxpg.cluster.dumpStats' mean-popcount line. 0 = the real test, 1 = drop the
    // facing gate, 2 = drop the occlusion ray, 3 = report only which
    // representatives landed on a surface at all, 4 = structural self-test of the
    // mask packing (every live lane votes visible, so the line MUST read 128.0 on
    // any scene with a lit voxel; see FINGERPRINT_PROBE_PACKING in the shader).
    // A representative that lands on the sky is not a receiver: it contributes no
    // bit to any voxel's fingerprint, so the 128-bit signature silently narrows.
    // Off restores the ported shape (one blind pick per cell) for A/B.
    AutoCVarInt g_fingerprintRetryPresample("vxpg.fingerprint.retryPresample",
        "Resample a representative until it lands on a surface instead of spending it on the sky",
        1, CVarFlags::EditCheckbox);

    AutoCVarInt g_fingerprintProbe("vxpg.fingerprint.probe",
        "Fingerprint visibility probe: 0 = normal, 1 = no facing gate, 2 = no occlusion, 3 = receiver validity, 4 = packing self-test",
        0, CVarFlags::EditDrag, 0, 4);

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

void VxpgFingerprintPass::CreateRootSignatures()
{
    {
        m_presampleRootSig = RootSignatureBuilder(L"VxpgFingerprint Presample RootSig", /*tableCount*/ 1)
                                 .Add(kPresampleSlots)
                                 .Build(m_device.Get());
    }

    {
        m_visibilityRootSig = RootSignatureBuilder(L"VxpgFingerprint Visibility RootSig", /*tableCount*/ 0)
                                  .Add(kVisibilitySlots)
                                  .Build(m_device.Get());
    }
}

void VxpgFingerprintPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_presampleProgram = cache.GetOrCreateCompute(m_device.Get(), m_presampleRootSig.Get(),
        "resources/shaders/vxpgFingerprint.presample.shader", L"VxpgFingerprint Presample PSO");
    m_visibilityProgram = cache.GetOrCreateCompute(m_device.Get(), m_visibilityRootSig.Get(),
        "resources/shaders/vxpgFingerprint.visibility.shader", L"VxpgFingerprint Visibility PSO");
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

bool VxpgFingerprintPass::IsRunnable()
{
    if (!m_initialized || !m_buildPass || !m_injectionPass || m_width == 0 || m_height == 0)
        return false;

    // The G-buffer this pass stratifies over is the injection pass's, bound out of
    // the global heap slot injection itself writes — so the only thing left to
    // check is that it exists.
    return m_injectionPass->GetShadingPointsTexture() != nullptr;
}

// Kernel 1: pick 128 stratified screen representatives and emit the dispatch args
// the visibility kernel and the cluster pass are then dispatched off.
void VxpgFingerprintPass::RunPresample(uint32_t frameIndex)
{
    if (!IsRunnable())
        return;

    auto* cmd = m_commandList.Get();

    GlobalDescriptorHeap& globalHeap = GlobalDescriptorHeap::Get();
    ID3D12DescriptorHeap* heaps[] = { globalHeap.GetHeap() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    cmd->SetComputeRootSignature(m_presampleRootSig.Get());
    uint32_t presampleConstants[4] = { m_width, m_height,
        PcgHash(frameIndex), g_fingerprintRetryPresample.Get() != 0 ? 1u : 0u };
    m_presampleRootSig.SetConstants(cmd, kPresampleConstants, presampleConstants, 4);
    m_presampleRootSig.SetTable(cmd, 0, globalHeap.GpuStart());
    m_presampleRootSig.Set(cmd, kPresampleRepresentatives, m_screenRepresentativePoints->GetGPUVirtualAddress());
    m_presampleRootSig.Set(cmd, kPresampleDispatchArgs, m_guidingDispatchArgs->GetGPUVirtualAddress());
    m_presampleRootSig.Set(cmd, kPresampleCounters, m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());

    cmd->SetPipelineState(m_presampleProgram->GetPipelineState());
    CommandContext::Get().Dispatch(1, 1, 1); // one 16x8 group = 128 representatives
}

// Kernel 2: fingerprint every lit voxel via inline shadow rays. Dispatched
// indirectly off gGuidingDispatchArgs[2] = (4, ceil(litVoxelCount/8), 1): X = 4
// groups of 32 = 128 representatives; Y sized to the live lit-voxel count.
// Replaces the worst-case (4, ceil(CAPACITY/8)=16384, 1) fixed dispatch
// (ADR 0003 option b).
//
// The args buffer's UNORDERED_ACCESS <-> INDIRECT_ARGUMENT flip is no longer done
// here: this node declares the buffer as IndirectArgument and the presample node
// declares it as a write, so the graph synthesizes both transitions.
void VxpgFingerprintPass::RunVisibility(D3D12_GPU_VIRTUAL_ADDRESS tlasVa)
{
    if (!IsRunnable() || tlasVa == 0)
        return;

    // No descriptor heap: this signature is all root descriptors (tableCount 0).
    auto* cmd = m_commandList.Get();

    cmd->SetComputeRootSignature(m_visibilityRootSig.Get());
    const uint32_t visibilityConstants[4] = { static_cast<uint32_t>(std::clamp(g_fingerprintProbe.Get(), 0, 3)), 0u, 0u, 0u };
    m_visibilityRootSig.SetConstants(cmd, kVisibilityConstants, visibilityConstants, 4);
    m_visibilityRootSig.Set(cmd, kVisibilityTlas, tlasVa);
    m_visibilityRootSig.Set(cmd, kVisibilityRepresentatives, m_screenRepresentativePoints->GetGPUVirtualAddress());
    m_visibilityRootSig.Set(cmd, kVisibilityLightPoints, m_buildPass->GetCompactVoxelLightPointsBuffer()->GetGPUVirtualAddress());
    m_visibilityRootSig.Set(cmd, kVisibilityDispatchArgs, m_guidingDispatchArgs->GetGPUVirtualAddress());
    m_visibilityRootSig.Set(cmd, kVisibilityFingerprints, m_voxelFingerprints->GetGPUVirtualAddress());
    cmd->SetPipelineState(m_visibilityProgram->GetPipelineState());

    constexpr uint32_t kVisibilityArgsOffset = 2 * sizeof(DirectX::XMUINT4); // entry [2]
    CommandContext::Get().DispatchIndirect(m_dispatchCommandSignature.Get(),
        m_guidingDispatchArgs->GetUnderlyingResource().Get(), kVisibilityArgsOffset);
}
