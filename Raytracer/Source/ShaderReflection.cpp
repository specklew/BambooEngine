#include "pch.h"
#include "ShaderReflection.h"

#include <d3d12shader.h>

#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
    D3D12_DESCRIPTOR_RANGE_TYPE RangeTypeForInputType(D3D_SHADER_INPUT_TYPE type)
    {
        switch (type)
        {
        case D3D_SIT_CBUFFER:
            return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        case D3D_SIT_SAMPLER:
            return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        case D3D_SIT_UAV_RWTYPED:
        case D3D_SIT_UAV_RWSTRUCTURED:
        case D3D_SIT_UAV_RWBYTEADDRESS:
        case D3D_SIT_UAV_APPEND_STRUCTURED:
        case D3D_SIT_UAV_CONSUME_STRUCTURED:
        case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
        case D3D_SIT_UAV_FEEDBACKTEXTURE:
            return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        default:
            // Textures, tbuffers, structured/byte-address buffers and acceleration
            // structures are all read-only views: t-registers.
            return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        }
    }

    // DXC reports an unbounded array as BindCount == 0.
    uint32_t RegisterCountFrom(UINT bindCount)
    {
        return bindCount == 0 ? ShaderReflection::kUnboundedRegisterCount : bindCount;
    }

    void AddUse(std::vector<ShaderResourceUse>& uses, const D3D12_SHADER_INPUT_BIND_DESC& desc)
    {
        const ShaderResourceUse use{desc.Name ? desc.Name : "<unnamed>", RangeTypeForInputType(desc.Type),
                                    desc.BindPoint, desc.Space, RegisterCountFrom(desc.BindCount)};

        for (ShaderResourceUse& existing : uses)
        {
            const bool sameRegister = existing.type == use.type && existing.baseRegister == use.baseRegister &&
                                      existing.registerSpace == use.registerSpace;
            if (!sameRegister)
                continue;

            // Two functions can bind the same register with different array
            // lengths; the signature has to satisfy the widest of them.
            if (use.registerCount == ShaderReflection::kUnboundedRegisterCount ||
                (existing.registerCount != ShaderReflection::kUnboundedRegisterCount &&
                 use.registerCount > existing.registerCount))
            {
                existing.registerCount = use.registerCount;
            }
            return;
        }

        uses.push_back(use);
    }

    // Deliberately never released: DXC's COM objects outlive static destruction
    // order badly, and this one is created once for the process (ADR 0018 §A7).
    IDxcUtils* GetDxcUtils()
    {
        static IDxcUtils* utils = []
        {
            IDxcUtils* created = nullptr;
            ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&created)));
            return created;
        }();
        return utils;
    }

    DxcBuffer BufferFrom(IDxcBlob* blob)
    {
        DxcBuffer buffer{};
        buffer.Ptr      = blob->GetBufferPointer();
        buffer.Size     = blob->GetBufferSize();
        buffer.Encoding = 0;
        return buffer;
    }

    D3D12_DESCRIPTOR_RANGE_TYPE RangeTypeForBindingKind(BindingKind kind)
    {
        switch (kind)
        {
        case BindingKind::Uav:     return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        case BindingKind::Srv:     return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        case BindingKind::Sampler: return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        default:                   return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        }
    }

    bool CoversRegister(const BindingSlot& slot, const ShaderResourceUse& use)
    {
        if (RangeTypeForBindingKind(slot.kind) != use.type || slot.registerSpace != use.registerSpace)
            return false;
        if (use.baseRegister < slot.shaderRegister)
            return false;
        if (slot.registerCount == kUnboundedRegisterCount)
            return true;
        if (use.registerCount == ShaderReflection::kUnboundedRegisterCount)
            return false; // an unbounded array needs an unbounded range
        return use.baseRegister + use.registerCount <= slot.shaderRegister + slot.registerCount;
    }

    char RegisterPrefix(D3D12_DESCRIPTOR_RANGE_TYPE type)
    {
        switch (type)
        {
        case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:     return 'u';
        case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:     return 't';
        case D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER: return 's';
        default:                                  return 'b';
        }
    }

    std::string RegisterNameOf(const ShaderResourceUse& use)
    {
        const std::string space = use.registerSpace ? fmt::format(",space{}", use.registerSpace) : std::string();
        return fmt::format("{}{}{}", RegisterPrefix(use.type), use.baseRegister, space);
    }
}

bool ShaderReflection::ReflectShaderUses(IDxcBlob* reflectionBlob, std::vector<ShaderResourceUse>& outUses)
{
    if (!reflectionBlob)
        return false;

    const DxcBuffer            buffer = BufferFrom(reflectionBlob);
    ComPtr<ID3D12ShaderReflection> reflection;
    if (FAILED(GetDxcUtils()->CreateReflection(&buffer, IID_PPV_ARGS(&reflection))))
        return false;

    D3D12_SHADER_DESC shaderDesc{};
    if (FAILED(reflection->GetDesc(&shaderDesc)))
        return false;

    for (uint32_t i = 0; i < shaderDesc.BoundResources; ++i)
    {
        D3D12_SHADER_INPUT_BIND_DESC bindDesc{};
        if (SUCCEEDED(reflection->GetResourceBindingDesc(i, &bindDesc)))
            AddUse(outUses, bindDesc);
    }

    return true;
}

bool ShaderReflection::ReflectLibraryUses(IDxcBlob* reflectionBlob, std::vector<ShaderResourceUse>& outUses)
{
    if (!reflectionBlob)
        return false;

    const DxcBuffer                 buffer = BufferFrom(reflectionBlob);
    ComPtr<ID3D12LibraryReflection> reflection;
    if (FAILED(GetDxcUtils()->CreateReflection(&buffer, IID_PPV_ARGS(&reflection))))
        return false;

    D3D12_LIBRARY_DESC libraryDesc{};
    if (FAILED(reflection->GetDesc(&libraryDesc)))
        return false;

    for (uint32_t functionIndex = 0; functionIndex < libraryDesc.FunctionCount; ++functionIndex)
    {
        ID3D12FunctionReflection* function = reflection->GetFunctionByIndex(static_cast<INT>(functionIndex));
        if (!function)
            continue;

        D3D12_FUNCTION_DESC functionDesc{};
        if (FAILED(function->GetDesc(&functionDesc)))
            continue;

        for (uint32_t i = 0; i < functionDesc.BoundResources; ++i)
        {
            D3D12_SHADER_INPUT_BIND_DESC bindDesc{};
            if (SUCCEEDED(function->GetResourceBindingDesc(i, &bindDesc)))
                AddUse(outUses, bindDesc);
        }
    }

    return true;
}

uint32_t ShaderReflection::ValidateAgainstRootSignature(const std::vector<ShaderResourceUse>& uses,
                                                        ID3D12RootSignature* rootSignature,
                                                        const std::string& shaderLabel)
{
    auto&                      library = RootSignatureLibrary::Get();
    const RootSignatureLayout* layout  = library.FindLayout(rootSignature);
    if (!layout)
        return 0;

    uint32_t problems = 0;
    for (const ShaderResourceUse& use : uses)
    {
        const std::vector<BindingSlot>& slots = layout->Slots();

        bool covered = false;
        for (size_t index = 0; index < slots.size(); ++index)
        {
            if (!CoversRegister(slots[index], use))
                continue;

            library.MarkSlotReferenced(rootSignature, index);
            covered = true;

            // Registers match. Names disagreeing means the two sides think this
            // register holds different resources — the signature still binds
            // something, so nothing crashes and the image is merely wrong. Only
            // the names catch it (ADR 0019 P3).
            //
            // Only for single-register slots: a range covering several registers
            // carries the name of its first, and the rest are named differently
            // on purpose (a bindless array, or a run of related UAVs).
            if (slots[index].registerCount == 1 && slots[index].name != use.name)
            {
                ++problems;
                spdlog::error("[ShaderBindings] {}: {} is '{}' in the shader but '{}' in {}", shaderLabel,
                              RegisterNameOf(use), use.name, slots[index].name, library.FindName(rootSignature));
            }
            break;
        }

        if (covered)
            continue;

        ++problems;
        spdlog::error("[ShaderBindings] {}: reads {} at {} which {} does not provide", shaderLabel, use.name,
                      RegisterNameOf(use), library.FindName(rootSignature));
    }

    return problems;
}

void ShaderReflection::ValidateShaderAsset(const char* shaderAssetPath, ID3D12RootSignature* rootSignature)
{
    auto&       resourceManager = ResourceManager::Get();
    const auto& shader = resourceManager.shaders.GetResource(resourceManager.GetOrLoadShader(AssetId(shaderAssetPath)));

    std::vector<ShaderResourceUse> uses;
    if (ReflectShaderUses(shader.reflection.Get(), uses))
        ValidateAgainstRootSignature(uses, rootSignature, shaderAssetPath);
}

void ShaderReflection::ValidateLibraryAsset(const char* shaderAssetPath, ID3D12RootSignature* rootSignature)
{
    auto&       resourceManager = ResourceManager::Get();
    const auto& shader = resourceManager.shaders.GetResource(resourceManager.GetOrLoadShader(AssetId(shaderAssetPath)));

    std::vector<ShaderResourceUse> uses;
    if (ReflectLibraryUses(shader.reflection.Get(), uses))
        ValidateAgainstRootSignature(uses, rootSignature, shaderAssetPath);
}
