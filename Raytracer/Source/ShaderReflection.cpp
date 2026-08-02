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

    bool CoversRegister(const RootSignatureBinding& binding, const ShaderResourceUse& use)
    {
        if (binding.type != use.type || binding.registerSpace != use.registerSpace)
            return false;
        if (use.baseRegister < binding.baseRegister)
            return false;
        if (binding.registerCount == RootSignatureLibrary::kUnboundedRegisterCount)
            return true;
        if (use.registerCount == ShaderReflection::kUnboundedRegisterCount)
            return false; // an unbounded array needs an unbounded range
        return use.baseRegister + use.registerCount <= binding.baseRegister + binding.registerCount;
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
                                                        ID3D12RootSignature*                 rootSignature,
                                                        const std::string&                   shaderLabel)
{
    auto&                                    library = RootSignatureLibrary::Get();
    const std::vector<RootSignatureBinding>* layout  = library.FindLayout(rootSignature);
    if (!layout)
        return 0;

    uint32_t uncovered = 0;
    for (const ShaderResourceUse& use : uses)
    {
        bool covered = false;
        for (size_t index = 0; index < layout->size(); ++index)
        {
            if (!CoversRegister((*layout)[index], use))
                continue;

            library.MarkBindingReferenced(rootSignature, index);
            covered = true;
            break;
        }

        if (covered)
            continue;

        ++uncovered;
        spdlog::error("[ShaderBindings] {}: reads {} at {}{}{} which {} does not provide", shaderLabel, use.name,
                      RegisterPrefix(use.type), use.baseRegister,
                      use.registerSpace ? fmt::format(",space{}", use.registerSpace) : std::string(),
                      library.FindName(rootSignature));
    }

    return uncovered;
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
