#pragma once
#include <string>
#include <vector>

// One register a shader actually references, as DXC reports it.
struct ShaderResourceUse
{
    std::string                 name;
    D3D12_DESCRIPTOR_RANGE_TYPE type          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    uint32_t                    baseRegister  = 0;
    uint32_t                    registerSpace = 0;
    uint32_t                    registerCount = 1; // kUnboundedRegisterCount for an unbounded array
};

// ADR 0017 phase 4: the binding declarations are hand-written, and nothing else
// checks them. A register a shader reads but the root signature never provides is
// an undefined read; a register the signature provides but no shader touches is a
// wasted root slot. Both are visible in DXC's reflection data, so ask it.
namespace ShaderReflection
{
    inline constexpr uint32_t kUnboundedRegisterCount = ~0u;

    // Compute/vertex/pixel DXIL. False when the blob carries no reflection data.
    bool ReflectShaderUses(IDxcBlob* reflectionBlob, std::vector<ShaderResourceUse>& outUses);

    // lib_6_x DXIL: the union over every exported function, because the global root
    // signature has to satisfy all of them at once. DXC's library-level bind counts
    // disagree with the per-function walk (DXC issue #2184), so this walks functions.
    bool ReflectLibraryUses(IDxcBlob* reflectionBlob, std::vector<ShaderResourceUse>& outUses);

    // Logs every use the signature does not cover. Returns how many it found, and
    // records which of the signature's bindings were referenced so that never-used
    // root parameters can be reported once everything has been built.
    uint32_t ValidateAgainstRootSignature(const std::vector<ShaderResourceUse>& uses,
                                          ID3D12RootSignature* rootSignature, const std::string& shaderLabel);

    // Loads (or reuses) a compiled shader asset and validates it in one call.
    // Every PSO the engine builds should run through one of these, otherwise its
    // root signature looks unreferenced when the unused-binding report runs.
    void ValidateShaderAsset(const char* shaderAssetPath, ID3D12RootSignature* rootSignature);
    void ValidateLibraryAsset(const char* shaderAssetPath, ID3D12RootSignature* rootSignature);
}
