#include "pch.h"
#include "RootSignatureLibrary.h"

#include "FrameBindingLayout.h"
#include "Renderer.h" // GetStaticSamplers
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
    D3D12_DESCRIPTOR_RANGE_TYPE RangeTypeOf(BindingKind kind)
    {
        switch (kind)
        {
        case BindingKind::Cbv:     return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        case BindingKind::Uav:     return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        case BindingKind::Sampler: return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        default:                   return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        }
    }

    char RegisterPrefix(BindingKind kind)
    {
        switch (kind)
        {
        case BindingKind::Uav:     return 'u';
        case BindingKind::Srv:     return 't';
        case BindingKind::Sampler: return 's';
        default:                   return 'b';
        }
    }

    std::string RegisterName(const BindingSlot& slot)
    {
        const std::string space = slot.registerSpace ? fmt::format(",space{}", slot.registerSpace) : std::string();
        if (slot.registerCount == kUnboundedRegisterCount)
            return fmt::format("{}{}{}[]", RegisterPrefix(slot.kind), slot.shaderRegister, space);
        if (slot.registerCount > 1)
            return fmt::format("{}{}{}[{}]", RegisterPrefix(slot.kind), slot.shaderRegister, space, slot.registerCount);
        return fmt::format("{}{}{}", RegisterPrefix(slot.kind), slot.shaderRegister, space);
    }
}

RootSignatureLibrary& RootSignatureLibrary::Get()
{
    static RootSignatureLibrary instance;
    return instance;
}

ComPtr<ID3D12RootSignature> RootSignatureLibrary::Create(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& desc,
                                                         const wchar_t* debugName, bool usesFrameLayout)
{
    assert(device && "Root signature needs a device");

    ComPtr<ID3DBlob> serialized, errors;
    const HRESULT    hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(hr))
    {
        spdlog::error("Root signature {} failed to serialize (0x{:08x}): {}",
                      debugName ? ConvertWcharToString(debugName) : "<unnamed>", static_cast<uint32_t>(hr),
                      errors ? static_cast<const char*>(errors->GetBufferPointer()) : "no diagnostic");
        ThrowIfFailed(hr);
    }

    ComPtr<ID3D12RootSignature> signature;
    ThrowIfFailed(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&signature)));
    if (debugName)
        signature->SetName(debugName);

    // The slot table arrives separately via AttachLayout: a signature built by
    // the builder gets one, the three empty DXR local signatures legitimately
    // have none to give.
    Entry entry;
    entry.signature       = signature;
    entry.usesFrameLayout = usesFrameLayout;
    entry.debugName       = debugName ? ConvertWcharToString(debugName) : "<unnamed>";
    m_signatures[signature.Get()] = std::move(entry);

    return signature;
}

const RootSignatureLayout* RootSignatureLibrary::FindLayout(ID3D12RootSignature* signature) const
{
    const auto it = m_signatures.find(signature);
    return it == m_signatures.end() ? nullptr : &it->second.layout;
}

std::string RootSignatureLibrary::FindName(ID3D12RootSignature* signature) const
{
    const auto it = m_signatures.find(signature);
    return it == m_signatures.end() ? std::string("<unregistered>") : it->second.debugName;
}

void RootSignatureLibrary::MarkSlotReferenced(ID3D12RootSignature* signature, size_t slotIndex)
{
    const auto it = m_signatures.find(signature);
    if (it != m_signatures.end() && slotIndex < it->second.slotReferenced.size())
        it->second.slotReferenced[slotIndex] = true;
}

void RootSignatureLibrary::LogUnreferencedSlots() const
{
    for (const auto& [signature, entry] : m_signatures)
    {
        std::string unreferenced;
        const std::vector<BindingSlot>& slots = entry.layout.Slots();
        for (size_t index = 0; index < slots.size(); ++index)
        {
            if (entry.slotReferenced[index])
                continue;

            const BindingSlot& slot = slots[index];
            if (slot.kind == BindingKind::Sampler)
                continue; // shared boilerplate, not a per-pass declaration
            if (entry.usesFrameLayout &&
                FrameBindingLayout::IsFrameRegister(RangeTypeOf(slot.kind), slot.shaderRegister, slot.registerSpace))
                continue;

            unreferenced += fmt::format(" {} ({})", slot.name, RegisterName(slot));
        }

        if (!unreferenced.empty())
            spdlog::info("[RootSignature] {}: no shader references{}", entry.debugName, unreferenced);
    }
}

std::string RootSignatureLibrary::DumpRootSignatures() const
{
    std::string dump;
    for (const auto& [signature, entry] : m_signatures)
    {
        const uint32_t cost = entry.layout.CostInDwords();
        dump += fmt::format("  {} — {} slots, {}/{} DWORDs{}\n", entry.debugName, entry.layout.Slots().size(), cost,
                            kMaxRootSignatureDwords, entry.layout.IsGraphics() ? ", graphics" : "");

        const std::vector<BindingSlot>& slots = entry.layout.Slots();
        for (size_t index = 0; index < slots.size(); ++index)
        {
            const BindingSlot& slot = slots[index];
            const char* storage = slot.storage == BindingStorage::Table           ? "table"
                                  : slot.storage == BindingStorage::RootConstants ? "constants"
                                  : slot.storage == BindingStorage::StaticSampler ? "sampler"
                                                                                  : "root";
            dump += fmt::format("      {:<10} {:<34} {:<9} {}\n", RegisterName(slot), slot.name, storage,
                                entry.slotReferenced[index] ? "" : "UNREFERENCED");
        }
    }
    return dump;
}

const RootSignatureLayout* RootSignatureLibrary::AttachLayout(ID3D12RootSignature* signature, RootSignatureLayout&& layout)
{
    const auto it = m_signatures.find(signature);
    assert(it != m_signatures.end() && "Layout attached to a signature the library did not create");
    it->second.layout = std::move(layout);
    it->second.slotReferenced.assign(it->second.layout.Slots().size(), false);
    return &it->second.layout;
}

uint32_t RootSignatureLayout::CostInDwords() const
{
    // One DWORD per table parameter regardless of how many slots ride in it;
    // two per root descriptor; one per 32-bit constant. Static samplers are
    // baked into the signature and cost no root space.
    uint32_t dwords = m_tableCount;
    for (const BindingSlot& slot : m_slots)
    {
        if (slot.storage == BindingStorage::RootConstants)
            dwords += slot.constantCount;
        else if (slot.storage == BindingStorage::RootDescriptor)
            dwords += 2;
    }
    return dwords;
}

// ---------------------------------------------------------------------------
// Slot-based construction and binding
// ---------------------------------------------------------------------------

uint32_t RootSignatureLayout::RootParameterOf(const BindingSlot& slot) const
{
    if (slot.registerSpace < kLookupSpaces && slot.shaderRegister < kLookupRegisters)
    {
        const uint8_t found = m_lookup[slot.registerSpace][static_cast<uint32_t>(slot.kind)][slot.shaderRegister];
        if (found != kNoRootParameter)
            return found;
    }

    spdlog::error("[RootSignature] {}: '{}' is not a root parameter of this signature", m_debugName, slot.name);
    assert(false && "Binding a slot through a signature that does not declare it");
    return 0;
}

void RootSignature::SetTable(ID3D12GraphicsCommandList* commandList, uint32_t tableIndex, D3D12_GPU_DESCRIPTOR_HANDLE heapStart) const
{
    assert(m_layout && "Binding through an unbuilt root signature");
    if (m_layout->IsGraphics())
        commandList->SetGraphicsRootDescriptorTable(tableIndex, heapStart);
    else
        commandList->SetComputeRootDescriptorTable(tableIndex, heapStart);
}

void RootSignature::Set(ID3D12GraphicsCommandList* commandList, const BindingSlot& slot, D3D12_GPU_VIRTUAL_ADDRESS address) const
{
    assert(m_layout && "Binding through an unbuilt root signature");
    const uint32_t index = m_layout->RootParameterOf(slot);

    if (m_layout->IsGraphics())
    {
        switch (slot.kind)
        {
        case BindingKind::Cbv: commandList->SetGraphicsRootConstantBufferView(index, address); break;
        case BindingKind::Srv: commandList->SetGraphicsRootShaderResourceView(index, address); break;
        default:               commandList->SetGraphicsRootUnorderedAccessView(index, address); break;
        }
        return;
    }

    switch (slot.kind)
    {
    case BindingKind::Cbv: commandList->SetComputeRootConstantBufferView(index, address); break;
    case BindingKind::Srv: commandList->SetComputeRootShaderResourceView(index, address); break;
    default:               commandList->SetComputeRootUnorderedAccessView(index, address); break;
    }
}

void RootSignature::SetConstants(ID3D12GraphicsCommandList* commandList, const BindingSlot& slot, const void* values, uint32_t numValues) const
{
    assert(m_layout && "Binding through an unbuilt root signature");
    const uint32_t index = m_layout->RootParameterOf(slot);

    if (m_layout->IsGraphics())
        commandList->SetGraphicsRoot32BitConstants(index, numValues, values, 0);
    else
        commandList->SetComputeRoot32BitConstants(index, numValues, values, 0);
}

RootSignatureBuilder::RootSignatureBuilder(const wchar_t* debugName, uint32_t tableCount)
    : m_debugName(debugName ? debugName : L"<unnamed>")
    , m_tableCount(tableCount)
    , m_nextRootParameter(tableCount) // table parameters occupy [0, tableCount)
{
}

RootSignatureBuilder& RootSignatureBuilder::AddFrameLayout()
{
    assert(m_tableCount >= 1 && "The frame layout needs a table parameter");
    m_usesFrameLayout = true;
    for (const BindingSlot& slot : FrameBindingLayout::Slots())
        Add(slot);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::Add(const BindingSlot& slot)
{
    if (slot.storage == BindingStorage::Table)
    {
        assert(slot.tableIndex < m_tableCount && "Table slot outside the declared table count");
        m_rootParameterIndices.push_back(slot.tableIndex);
    }
    else if (slot.storage == BindingStorage::StaticSampler)
    {
        m_rootParameterIndices.push_back(UINT32_MAX); // not a root parameter
    }
    else
    {
        m_rootParameterIndices.push_back(m_nextRootParameter++);
    }

    m_slots.push_back(slot);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::Add(const BindingSlot* slots, size_t count)
{
    for (size_t index = 0; index < count; ++index)
        Add(slots[index]);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::WithStaticSamplers()
{
    if (m_staticSamplers)
        return *this;

    m_staticSamplers = true;
    // Declared as slots too, so a shader sampling through s0-s5 validates like
    // any other binding instead of looking like an undeclared read.
    for (const BindingSlot& sampler : FrameBindingLayout::StaticSamplerSlots())
        Add(sampler);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::WithFlags(D3D12_ROOT_SIGNATURE_FLAGS flags)
{
    m_flags = flags;
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::ForGraphics()
{
    m_graphics = true;
    return *this;
}

RootSignature RootSignatureBuilder::Build(ID3D12Device* device)
{
    // Descriptor ranges per table, in slot declaration order.
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> tableRanges(m_tableCount);
    std::vector<CD3DX12_ROOT_PARAMETER>              parameters(m_nextRootParameter);

    for (size_t index = 0; index < m_slots.size(); ++index)
    {
        const BindingSlot& slot = m_slots[index];

        if (slot.storage == BindingStorage::Table)
        {
            D3D12_DESCRIPTOR_RANGE range;
            range.RangeType                         = RangeTypeOf(slot.kind);
            range.BaseShaderRegister                = slot.shaderRegister;
            range.NumDescriptors                    = slot.registerCount;
            range.RegisterSpace                     = slot.registerSpace;
            range.OffsetInDescriptorsFromTableStart = slot.heapOffset;
            tableRanges[slot.tableIndex].push_back(range);
            continue;
        }

        // Static samplers are declared as slots so validation sees them, but the
        // desc takes them from Renderer::GetStaticSamplers(), not from a root
        // parameter — they have no index to fill.
        if (slot.storage == BindingStorage::StaticSampler)
            continue;

        const uint32_t parameterIndex = m_rootParameterIndices[index];
        if (slot.storage == BindingStorage::RootConstants)
        {
            parameters[parameterIndex].InitAsConstants(slot.constantCount, slot.shaderRegister, slot.registerSpace);
            continue;
        }

        switch (slot.kind)
        {
        case BindingKind::Cbv:
            parameters[parameterIndex].InitAsConstantBufferView(slot.shaderRegister, slot.registerSpace);
            break;
        case BindingKind::Srv:
            parameters[parameterIndex].InitAsShaderResourceView(slot.shaderRegister, slot.registerSpace);
            break;
        default:
            parameters[parameterIndex].InitAsUnorderedAccessView(slot.shaderRegister, slot.registerSpace);
            break;
        }
    }

    for (uint32_t tableIndex = 0; tableIndex < m_tableCount; ++tableIndex)
    {
        assert(!tableRanges[tableIndex].empty() && "Declared a table parameter that no slot fills");
        parameters[tableIndex].InitAsDescriptorTable(static_cast<UINT>(tableRanges[tableIndex].size()), tableRanges[tableIndex].data());
    }

    // Built unconditionally so the array outlives the desc; only referenced when asked for.
    const auto samplers      = Renderer::GetStaticSamplers();
    const UINT samplerCount  = m_staticSamplers ? static_cast<UINT>(samplers.size()) : 0;
    const auto* samplerDescs = m_staticSamplers ? samplers.data() : nullptr;

    CD3DX12_ROOT_SIGNATURE_DESC desc(static_cast<UINT>(parameters.size()), parameters.data(), samplerCount, samplerDescs, m_flags);

    RootSignature built;
    built.m_signature = RootSignatureLibrary::Get().Create(device, desc, m_debugName.c_str(), m_usesFrameLayout);

    RootSignatureLayout layout;
    layout.m_debugName = ConvertWcharToString(m_debugName.c_str());
    layout.m_graphics  = m_graphics;
    std::memset(layout.m_lookup, RootSignatureLayout::kNoRootParameter, sizeof(layout.m_lookup));

    for (size_t index = 0; index < m_slots.size(); ++index)
    {
        const BindingSlot& slot = m_slots[index];
        if (slot.storage == BindingStorage::Table || slot.storage == BindingStorage::StaticSampler)
            continue; // bound by table index or not bound at all; never looked up by register

        assert(slot.registerSpace < RootSignatureLayout::kLookupSpaces &&
               slot.shaderRegister < RootSignatureLayout::kLookupRegisters &&
               "Root descriptor outside the lookup table's range");
        assert(m_rootParameterIndices[index] < RootSignatureLayout::kNoRootParameter);
        layout.m_lookup[slot.registerSpace][static_cast<uint32_t>(slot.kind)][slot.shaderRegister] =
            static_cast<uint8_t>(m_rootParameterIndices[index]);
    }

    layout.m_tableCount           = m_tableCount;
    layout.m_slots                = std::move(m_slots);
    layout.m_rootParameterIndices = std::move(m_rootParameterIndices);

    // D3D12 rejects an oversized signature at serialization, but the message
    // does not say by how much or which parameters are to blame. Say so here,
    // where the slot table is still in hand.
    const uint32_t cost = layout.CostInDwords();
    if (cost > kMaxRootSignatureDwords)
    {
        spdlog::error("Root signature {} costs {} DWORDs, over the {} limit", layout.m_debugName, cost,
                      kMaxRootSignatureDwords);
        assert(false && "Root signature exceeds the 64-DWORD budget");
    }

    built.m_layout = RootSignatureLibrary::Get().AttachLayout(built.m_signature.Get(), std::move(layout));

    return built;
}
