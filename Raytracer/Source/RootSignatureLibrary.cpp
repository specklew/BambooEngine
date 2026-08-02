#include "pch.h"
#include "RootSignatureLibrary.h"

#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
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

ComPtr<ID3D12RootSignature> RootSignatureLibrary::Create(ID3D12Device*                     device,
                                                         const D3D12_ROOT_SIGNATURE_DESC& desc,
                                                         const wchar_t*                   debugName)
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
    entry.signature = signature;
    entry.debugName = debugName ? ConvertWcharToString(debugName) : "<unnamed>";
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
