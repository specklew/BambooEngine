#include "pch.h"
#include "Utils/GpuMemoryReport.h"
#include "CommandContext.h"
#include "SuperpixelBuildPass.h"

#include "Constants.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
    // The seven SLIC buffers, off the global heap. Declared one per register
    // rather than as a single seven-wide range so each carries its own name for
    // reflection validation; the kernels differ only in which they touch.
    //
    // These lived in a private heap until ADR 0017 phase 6a. Every one of them
    // already had a global slot, written when its resource is created — and the
    // private copy of the ShadingPoints descriptor did not, so it was re-pointed
    // lazily at record time instead. That was sound only while the frame ended in
    // a full flush: without one, the rewrite lands in a heap that frames still in
    // flight are reading, and between injection recreating ShadingPoints and the
    // next bind the private descriptor named a destroyed resource. Both disappear
    // with the heap, which is what GPU-based validation caught.
    constexpr BindingSlot kSuperpixelConstants = PassRootConstants("SuperpixelCB", SUPERPIXEL_REG_CB, 12);
    constexpr BindingSlot kSuperpixelSlots[] = {
        PassTableEntry("u_input", BindingKind::Uav, SUPERPIXEL_REG_INPUT, GlobalDescriptor::ShadingPoints),
        PassTableEntry("u_center", BindingKind::Uav, SUPERPIXEL_REG_CENTER, GlobalDescriptor::SuperpixelCenter),
        PassTableEntry("u_index", BindingKind::Uav, SUPERPIXEL_REG_INDEX, GlobalDescriptor::SuperpixelIndex),
        PassTableEntry("u_spixel_counter", BindingKind::Uav, SUPERPIXEL_REG_SPIXEL_COUNTER, GlobalDescriptor::SpixelCounter),
        PassTableEntry("u_spixel_gathered", BindingKind::Uav, SUPERPIXEL_REG_SPIXEL_GATHERED, GlobalDescriptor::SpixelGathered),
        PassTableEntry("u_fuzzy_weight", BindingKind::Uav, SUPERPIXEL_REG_FUZZY_WEIGHT, GlobalDescriptor::FuzzyWeight),
        PassTableEntry("u_fuzzy_index", BindingKind::Uav, SUPERPIXEL_REG_FUZZY_INDEX, GlobalDescriptor::FuzzyIndex),
    };

    constexpr uint32_t SP = Constants::Graphics::SUPERPIXEL_SIZE;

    // (1 / (1.4242 * spixel_size))^2 — squared screen-xy normalizer, as in SIByL.
    float ComputeMaxXyDistSquared()
    {
        float d = 1.0f / (1.4242f * static_cast<float>(SP));
        return d * d;
    }

    struct SuperpixelConstants
    {
        int32_t  mapX, mapY;
        int32_t  imgW, imgH;
        int32_t  spixelSize;
        float    weight;
        float    maxXyDist;
        float    maxColorDist;
        uint32_t writeGather;
        uint32_t pad0, pad1, pad2;
    };
    static_assert(sizeof(SuperpixelConstants) == 12 * sizeof(uint32_t), "root constant count mismatch");
}

void SuperpixelBuildPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    spdlog::info("Initializing superpixel build pass...");
    m_device      = device;
    m_commandList = commandList;

    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void SuperpixelBuildPass::CreateRootSignature()
{
    m_rootSig = RootSignatureBuilder(L"SuperpixelBuild RootSig", /*tableCount*/ 1)
                    .Add(kSuperpixelConstants)
                    .Add(kSuperpixelSlots)
                    .Build(m_device.Get());
}

void SuperpixelBuildPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_initProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/superpixelBuild.initSeed.shader", L"Superpixel InitSeed PSO");
    m_assocProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/superpixelBuild.findAssoc.shader", L"Superpixel FindAssoc PSO");
    m_sumProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/superpixelBuild.sumCenter.shader", L"Superpixel SumCenter PSO");
    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/superpixelBuild.clearCounter.shader", L"Superpixel ClearCounter PSO");
}

void SuperpixelBuildPass::CreateBuffers()
{
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    auto makeTex = [&](DXGI_FORMAT fmt, uint32_t w, uint32_t h, const wchar_t* name, ComPtr<ID3D12Resource>& out)
    {
        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(
            fmt, w, h, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&out)));
        out->SetName(name);
    };

    makeTex(DXGI_FORMAT_R32G32B32A32_FLOAT, m_mapX,   m_mapY,   L"Superpixel Center",  m_center);
    makeTex(DXGI_FORMAT_R32_SINT,           m_width,  m_height, L"Superpixel Index",   m_index);
    makeTex(DXGI_FORMAT_R32_UINT,           m_mapX,   m_mapY,   L"Superpixel Counter", m_counter);
    makeTex(DXGI_FORMAT_R32G32_SINT,        m_mapX * SP, m_mapY * SP, L"Superpixel Gathered", m_gathered);
    makeTex(DXGI_FORMAT_R32G32B32A32_FLOAT, m_width,  m_height, L"Superpixel FuzzyWeight", m_fuzzyWeight);
    makeTex(DXGI_FORMAT_R32G32B32A32_SINT,  m_width,  m_height, L"Superpixel FuzzyIndex",  m_fuzzyIndex);
}

void SuperpixelBuildPass::OnResize(uint32_t width, uint32_t height, ID3D12Resource* shadingPoints)
{
    if (!m_initialized || width == 0 || height == 0 || !shadingPoints)
        return;

    m_width  = width;
    m_height = height;
    m_mapX   = (width  + SP - 1) / SP;
    m_mapY   = (height + SP - 1) / SP;

    m_center.Reset(); m_index.Reset(); m_counter.Reset(); m_gathered.Reset();
    m_fuzzyWeight.Reset(); m_fuzzyIndex.Reset();

    CreateBuffers();
    m_buffersCreated = true;

    // The caller writes the new resources into their global-heap slots
    // (Renderer::WriteSuperpixelUavsToGlobalHeap and its cluster-visibility
    // counterpart, which owns the gathered/counter pair) — always at a flush
    // point, which is what makes a shader-visible descriptor safe to overwrite.
}

void SuperpixelBuildPass::WriteIndexUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    if (!m_index) return;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format        = DXGI_FORMAT_R32_SINT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_index.Get(), nullptr, &uav, dest);
}

void SuperpixelBuildPass::WriteCenterUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    if (!m_center) return;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format        = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_center.Get(), nullptr, &uav, dest);
}

void SuperpixelBuildPass::WriteFuzzyWeightUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    if (!m_fuzzyWeight) return;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format        = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_fuzzyWeight.Get(), nullptr, &uav, dest);
}

void SuperpixelBuildPass::WriteFuzzyIndexUavTo(D3D12_CPU_DESCRIPTOR_HANDLE dest) const
{
    if (!m_fuzzyIndex) return;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format        = DXGI_FORMAT_R32G32B32A32_SINT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_fuzzyIndex.Get(), nullptr, &uav, dest);
}

void SuperpixelBuildPass::SetFrameInputs(float weight, float posNormalizer)
{
    m_weight        = weight;
    m_posNormalizer = posNormalizer;
}

// Binds the private heap, root signature and constants. Every kernel calls it:
// separate nodes may have barriers placed between them, so none may inherit
// another's root state. writeGather selects the association variant.
bool SuperpixelBuildPass::BindCommon(bool writeGather)
{
    if (!m_initialized || !m_buffersCreated)
        return false;

    auto* cmd = m_commandList.Get();

    ID3D12DescriptorHeap* heaps[] = { GlobalDescriptorHeap::Get().GetHeap() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);
    cmd->SetComputeRootSignature(m_rootSig.Get());
    m_rootSig.SetTable(cmd, 0, GlobalDescriptorHeap::Get().GpuStart());

    SuperpixelConstants c{};
    c.mapX = static_cast<int32_t>(m_mapX);
    c.mapY = static_cast<int32_t>(m_mapY);
    c.imgW = static_cast<int32_t>(m_width);
    c.imgH = static_cast<int32_t>(m_height);
    c.spixelSize   = static_cast<int32_t>(SP);
    c.weight       = m_weight;
    c.maxXyDist    = ComputeMaxXyDistSquared();
    c.maxColorDist = m_posNormalizer;
    c.writeGather  = writeGather ? 1u : 0u;
    m_rootSig.SetConstants(cmd, kSuperpixelConstants, &c, 12);

    return true;
}

// Seed centers from tile middle pixels.
void SuperpixelBuildPass::RunInitSeeds()
{
    if (!BindCommon(false))
        return;

    m_commandList->SetPipelineState(m_initProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_mapX + 7) / 8, (m_mapY + 7) / 8, 1);
}

// Assign each pixel to its nearest center. The final pass sets writeGather so it
// also emits the per-superpixel pixel lists, consistent with converged centers.
void SuperpixelBuildPass::RunAssociate(bool writeGather)
{
    if (!BindCommon(writeGather))
        return;

    m_commandList->SetPipelineState(m_assocProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);
}

// Average-update the centers from the current assignment.
void SuperpixelBuildPass::RunSumCenters()
{
    if (!BindCommon(false))
        return;

    m_commandList->SetPipelineState(m_sumProgram->GetPipelineState());
    CommandContext::Get().Dispatch(m_mapX, m_mapY, 1);
}

void SuperpixelBuildPass::RunClearCounter()
{
    if (!BindCommon(false))
        return;

    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_mapX + 7) / 8, (m_mapY + 7) / 8, 1);
}

// P5. Every one of these is screen-sized or superpixel-map-sized, so this stage tracks
// the render resolution and is untouched by the grid.
void SuperpixelBuildPass::ReportMemory(GpuMemoryReport& report) const
{
    using namespace GpuMemoryStage;
    report.Add(Superpixel, "superpixel centers",     m_center.Get());
    report.Add(Superpixel, "superpixel index",       m_index.Get());
    report.Add(Superpixel, "superpixel counter",     m_counter.Get());
    report.Add(Superpixel, "superpixel gathered",    m_gathered.Get());
    report.Add(Superpixel, "fuzzy weight",           m_fuzzyWeight.Get());
    report.Add(Superpixel, "fuzzy index",            m_fuzzyIndex.Get());
}
