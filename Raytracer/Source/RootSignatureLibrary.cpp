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

    D3D12_DESCRIPTOR_RANGE_TYPE RangeTypeForRootParameter(D3D12_ROOT_PARAMETER_TYPE type)
    {
        switch (type)
        {
        case D3D12_ROOT_PARAMETER_TYPE_SRV: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        case D3D12_ROOT_PARAMETER_TYPE_UAV: return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        default:                            return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        }
    }

    void FlattenRootParameters(const D3D12_ROOT_SIGNATURE_DESC&    desc,
                               std::vector<RootSignatureBinding>& outBindings)
    {
        for (uint32_t parameterIndex = 0; parameterIndex < desc.NumParameters; ++parameterIndex)
        {
            const D3D12_ROOT_PARAMETER& parameter = desc.pParameters[parameterIndex];

            if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            {
                const auto& table = parameter.DescriptorTable;
                for (uint32_t rangeIndex = 0; rangeIndex < table.NumDescriptorRanges; ++rangeIndex)
                {
                    const D3D12_DESCRIPTOR_RANGE& range = table.pDescriptorRanges[rangeIndex];
                    outBindings.push_back({range.RangeType, range.BaseShaderRegister, range.RegisterSpace,
                                           range.NumDescriptors == UINT_MAX
                                               ? RootSignatureLibrary::kUnboundedRegisterCount
                                               : range.NumDescriptors,
                                           parameterIndex, true});
                }
                continue;
            }

            // Root constants occupy a b-register exactly like a root CBV does.
            const uint32_t shaderRegister = parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
                                                ? parameter.Constants.ShaderRegister
                                                : parameter.Descriptor.ShaderRegister;
            const uint32_t registerSpace  = parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
                                                ? parameter.Constants.RegisterSpace
                                                : parameter.Descriptor.RegisterSpace;

            outBindings.push_back({RangeTypeForRootParameter(parameter.ParameterType), shaderRegister,
                                   registerSpace, 1, parameterIndex, false});
        }

        for (uint32_t samplerIndex = 0; samplerIndex < desc.NumStaticSamplers; ++samplerIndex)
        {
            const D3D12_STATIC_SAMPLER_DESC& sampler = desc.pStaticSamplers[samplerIndex];
            outBindings.push_back({D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, sampler.ShaderRegister,
                                   sampler.RegisterSpace, 1, UINT_MAX, false});
        }
    }
}

RootSignatureLibrary& RootSignatureLibrary::Get()
{
    static RootSignatureLibrary instance;
    return instance;
}

ComPtr<ID3D12RootSignature> RootSignatureLibrary::Create(ID3D12Device*                    device,
                                                         const D3D12_ROOT_SIGNATURE_DESC& desc,
                                                         const wchar_t*                   debugName,
                                                         bool                             usesFrameLayout)
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
    ThrowIfFailed(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&signature)));
    if (debugName)
        signature->SetName(debugName);

    Entry entry;
    entry.signature       = signature;
    entry.usesFrameLayout = usesFrameLayout;
    entry.debugName       = debugName ? ConvertWcharToString(debugName) : "<unnamed>";
    FlattenRootParameters(desc, entry.bindings);
    entry.bindingReferenced.assign(entry.bindings.size(), false);
    m_signatures[signature.Get()] = std::move(entry);

    return signature;
}

const std::vector<RootSignatureBinding>* RootSignatureLibrary::FindLayout(ID3D12RootSignature* signature) const
{
    const auto it = m_signatures.find(signature);
    return it == m_signatures.end() ? nullptr : &it->second.bindings;
}

std::string RootSignatureLibrary::FindName(ID3D12RootSignature* signature) const
{
    const auto it = m_signatures.find(signature);
    return it == m_signatures.end() ? std::string("<unregistered>") : it->second.debugName;
}

void RootSignatureLibrary::MarkBindingReferenced(ID3D12RootSignature* signature, size_t bindingIndex)
{
    const auto it = m_signatures.find(signature);
    if (it != m_signatures.end() && bindingIndex < it->second.bindingReferenced.size())
        it->second.bindingReferenced[bindingIndex] = true;
}

void RootSignatureLibrary::LogUnreferencedBindings() const
{
    for (const auto& [signature, entry] : m_signatures)
    {
        std::string unreferenced;
        for (size_t index = 0; index < entry.bindings.size(); ++index)
        {
            if (entry.bindingReferenced[index])
                continue;

            const RootSignatureBinding& binding = entry.bindings[index];
            if (binding.type == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER)
                continue; // static samplers are shared boilerplate, not a per-pass declaration
            if (entry.usesFrameLayout &&
                FrameBindingLayout::IsFrameRegister(binding.type, binding.baseRegister, binding.registerSpace))
                continue;

            const char prefix = binding.type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV   ? 'u'
                                : binding.type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV ? 't'
                                                                                  : 'b';
            const std::string space =
                binding.registerSpace ? fmt::format(",space{}", binding.registerSpace) : std::string();
            unreferenced += fmt::format(" {}{}{}", prefix, binding.baseRegister, space);
        }

        if (!unreferenced.empty())
            spdlog::info("[RootSignature] {}: no shader references{}", entry.debugName, unreferenced);
    }
}

const RootSignatureLayout* RootSignatureLibrary::AttachLayout(ID3D12RootSignature* signature,
                                                              RootSignatureLayout&& layout)
{
    const auto it = m_signatures.find(signature);
    assert(it != m_signatures.end() && "Layout attached to a signature the library did not create");
    it->second.slotLayout = std::move(layout);
    return &it->second.slotLayout;
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

void RootSignature::SetTable(ID3D12GraphicsCommandList* commandList, uint32_t tableIndex,
                             D3D12_GPU_DESCRIPTOR_HANDLE heapStart) const
{
    assert(m_layout && "Binding through an unbuilt root signature");
    if (m_layout->IsGraphics())
        commandList->SetGraphicsRootDescriptorTable(tableIndex, heapStart);
    else
        commandList->SetComputeRootDescriptorTable(tableIndex, heapStart);
}

void RootSignature::Set(ID3D12GraphicsCommandList* commandList, const BindingSlot& slot,
                        D3D12_GPU_VIRTUAL_ADDRESS address) const
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

void RootSignature::SetConstants(ID3D12GraphicsCommandList* commandList, const BindingSlot& slot,
                                 const void* values, uint32_t numValues) const
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
    m_staticSamplers = true;
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
        parameters[tableIndex].InitAsDescriptorTable(static_cast<UINT>(tableRanges[tableIndex].size()),
                                                     tableRanges[tableIndex].data());
    }

    // Built unconditionally so the array outlives the desc; only referenced when asked for.
    const auto samplers      = Renderer::GetStaticSamplers();
    const UINT samplerCount  = m_staticSamplers ? static_cast<UINT>(samplers.size()) : 0;
    const auto* samplerDescs = m_staticSamplers ? samplers.data() : nullptr;

    CD3DX12_ROOT_SIGNATURE_DESC desc(static_cast<UINT>(parameters.size()), parameters.data(), samplerCount,
                                     samplerDescs, m_flags);

    RootSignature built;
    built.m_signature = RootSignatureLibrary::Get().Create(device, desc, m_debugName.c_str(), m_usesFrameLayout);

    RootSignatureLayout layout;
    layout.m_debugName = ConvertWcharToString(m_debugName.c_str());
    layout.m_graphics  = m_graphics;
    std::memset(layout.m_lookup, RootSignatureLayout::kNoRootParameter, sizeof(layout.m_lookup));

    for (size_t index = 0; index < m_slots.size(); ++index)
    {
        const BindingSlot& slot = m_slots[index];
        if (slot.storage == BindingStorage::Table)
            continue; // bound by table index, never looked up by register

        assert(slot.registerSpace < RootSignatureLayout::kLookupSpaces &&
               slot.shaderRegister < RootSignatureLayout::kLookupRegisters &&
               "Root descriptor outside the lookup table's range");
        assert(m_rootParameterIndices[index] < RootSignatureLayout::kNoRootParameter);
        layout.m_lookup[slot.registerSpace][static_cast<uint32_t>(slot.kind)][slot.shaderRegister] =
            static_cast<uint8_t>(m_rootParameterIndices[index]);
    }

    layout.m_slots                = std::move(m_slots);
    layout.m_rootParameterIndices = std::move(m_rootParameterIndices);
    built.m_layout = RootSignatureLibrary::Get().AttachLayout(built.m_signature.Get(), std::move(layout));

    return built;
}
