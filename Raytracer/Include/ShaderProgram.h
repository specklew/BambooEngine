#pragma once
#include <string>
#include <unordered_map>

// ADR 0017 L4: a compute shader bound to a root signature, plus the PSO built
// from the pair. Programs are pointer-stable and owned by the cache, so a shader
// reload swaps the PSO underneath every pass that holds one — passes bind through
// GetPipelineState() and never rebuild anything themselves.
class ComputeProgram
{
public:
    [[nodiscard]] ID3D12PipelineState* GetPipelineState() const { return m_pipelineState.Get(); }
    [[nodiscard]] ID3D12RootSignature* GetRootSignature() const { return m_rootSignature.Get(); }

private:
    friend class ShaderProgramCache;

    // False on a compile/create failure, with the previous PSO left in place.
    bool Build();

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    ID3D12Device*                               m_device = nullptr;
    std::string                                 m_shaderAssetPath;
    std::wstring                                m_debugName;
};

class ShaderProgramCache
{
public:
    static ShaderProgramCache& Get();

    // Keyed by (root signature, shader path): the same shader under two root
    // signatures is two PSOs. The cache holds a reference to the root signature so
    // a program outliving its creator pass can still be rebuilt.
    ComputeProgram* GetOrCreateCompute(ID3D12Device* device, ID3D12RootSignature* rootSignature, const char* shaderAssetPath, const wchar_t* debugName);

    // Shader reload: recompilation already happened in ResourceManager, so this
    // just rebuilds every PSO from the fresh bytecode. The GPU must be idle.
    void RebuildAll();

    [[nodiscard]] size_t GetProgramCount() const { return m_computePrograms.size(); }

private:
    std::unordered_map<std::string, std::unique_ptr<ComputeProgram>> m_computePrograms;
};
