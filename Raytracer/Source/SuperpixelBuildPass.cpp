#include "pch.h"
#include "CommandContext.h"
#include "SuperpixelBuildPass.h"

#include "Constants.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
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
    // b0 = 12 root constants (SuperpixelConstants). Table = 7 UAVs u0..u6:
    // u0 ShadingPoints, u1 center, u2 index, u3 counter, u4 gathered,
    // u5 fuzzy weights, u6 fuzzy indices.
    CD3DX12_DESCRIPTOR_RANGE uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 7, 0);

    CD3DX12_ROOT_PARAMETER params[2];
    params[0].InitAsConstants(12, 0);
    params[1].InitAsDescriptorTable(1, &uavRange);

    CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    m_rootSig = RootSignatureLibrary::Get().Create(m_device.Get(), desc, L"SuperpixelBuild RootSig");
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

void SuperpixelBuildPass::CreatePrivateHeap(ID3D12Resource* shadingPoints)
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 7;
    heapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_privateHeap)));

    UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_privateHeap->GetCPUDescriptorHandleForHeapStart());

    auto writeUav = [&](ID3D12Resource* res, DXGI_FORMAT fmt)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
        uav.Format             = fmt;
        uav.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Texture2D.MipSlice = 0;
        m_device->CreateUnorderedAccessView(res, nullptr, &uav, handle);
        handle.Offset(1, inc);
    };

    writeUav(shadingPoints,   DXGI_FORMAT_R32G32B32A32_FLOAT); // u0 ShadingPoints
    writeUav(m_center.Get(),  DXGI_FORMAT_R32G32B32A32_FLOAT); // u1
    writeUav(m_index.Get(),   DXGI_FORMAT_R32_SINT);           // u2
    writeUav(m_counter.Get(), DXGI_FORMAT_R32_UINT);           // u3
    writeUav(m_gathered.Get(),DXGI_FORMAT_R32G32_SINT);        // u4
    writeUav(m_fuzzyWeight.Get(), DXGI_FORMAT_R32G32B32A32_FLOAT); // u5
    writeUav(m_fuzzyIndex.Get(),  DXGI_FORMAT_R32G32B32A32_SINT);  // u6

    m_boundShadingPoints = shadingPoints;
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
    m_privateHeap.Reset();

    CreateBuffers();
    CreatePrivateHeap(shadingPoints);
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

void SuperpixelBuildPass::SetFrameInputs(ID3D12Resource* shadingPoints, float weight, float posNormalizer)
{
    m_shadingPoints = shadingPoints;
    m_weight        = weight;
    m_posNormalizer = posNormalizer;
}

// Binds the private heap, root signature and constants. Every kernel calls it:
// separate nodes may have barriers placed between them, so none may inherit
// another's root state. writeGather selects the association variant.
bool SuperpixelBuildPass::BindCommon(bool writeGather)
{
    if (!m_initialized || !m_privateHeap)
        return false;

    // Injection may have recreated ShadingPoints (resize / scene change / shader
    // reload); re-point private-heap slot 0 so its descriptor never dangles. Only
    // happens at flush points, so overwriting the live descriptor is safe.
    if (m_shadingPoints && m_shadingPoints != m_boundShadingPoints)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
        uav.Format        = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(m_shadingPoints, nullptr, &uav,
            m_privateHeap->GetCPUDescriptorHandleForHeapStart()); // slot 0 = u_input
        m_boundShadingPoints = m_shadingPoints;
    }

    auto* cmd = m_commandList.Get();

    ID3D12DescriptorHeap* heaps[] = { m_privateHeap.Get() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);
    cmd->SetComputeRootSignature(m_rootSig.Get());
    cmd->SetComputeRootDescriptorTable(1, m_privateHeap->GetGPUDescriptorHandleForHeapStart());

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
    cmd->SetComputeRoot32BitConstants(0, 12, &c, 0);

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
