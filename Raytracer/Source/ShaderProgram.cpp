#include "pch.h"
#include "ShaderProgram.h"

#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "ShaderReflection.h"
#include "Utils/Utils.h"

// Builds into a temporary and only publishes on success: a hot reload that hits a
// shader with a typo must leave the pass bound to its last good PSO rather than to
// null. Returns false instead of throwing so one bad shader cannot take the rest
// of RebuildAll() with it.
bool ComputeProgram::Build()
{
    auto& resourceManager = ResourceManager::Get();
    auto  shaderHandle    = resourceManager.GetOrLoadShader(AssetId(m_shaderAssetPath));
    auto  bytecode        = resourceManager.shaders.GetResource(shaderHandle).bytecode;
    if (!bytecode)
    {
        spdlog::error("Compute program {}: no bytecode", m_shaderAssetPath);
        return false;
    }

    ShaderReflection::ValidateShaderAsset(m_shaderAssetPath.c_str(), m_rootSignature.Get());

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = m_rootSignature.Get();
    desc.CS             = CD3DX12_SHADER_BYTECODE(bytecode->GetBufferPointer(), bytecode->GetBufferSize());

    Microsoft::WRL::ComPtr<ID3D12PipelineState> rebuilt;
    const HRESULT hr = m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&rebuilt));
    if (FAILED(hr))
    {
        spdlog::error("Compute program {} failed to build (0x{:08x}); keeping the previous PSO",
                      m_shaderAssetPath, static_cast<uint32_t>(hr));
        return false;
    }

    rebuilt->SetName(m_debugName.c_str());
    m_pipelineState = std::move(rebuilt);
    return true;
}

ShaderProgramCache& ShaderProgramCache::Get()
{
    static ShaderProgramCache instance;
    return instance;
}

ComputeProgram* ShaderProgramCache::GetOrCreateCompute(ID3D12Device* device, ID3D12RootSignature* rootSignature,
                                                       const char* shaderAssetPath, const wchar_t* debugName)
{
    assert(device && rootSignature && shaderAssetPath && "Compute program needs a device, root signature and shader");

    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(rootSignature)) + '|' + shaderAssetPath;

    if (auto it = m_computePrograms.find(key); it != m_computePrograms.end())
        return it->second.get();

    auto program = std::make_unique<ComputeProgram>();
    program->m_device          = device;
    program->m_rootSignature   = rootSignature;
    program->m_shaderAssetPath = shaderAssetPath;
    program->m_debugName       = debugName ? debugName : L"Compute PSO";

    // The first build has no previous PSO to fall back on, so a failure here is
    // fatal in the same way it always was.
    if (!program->Build())
        ThrowIfFailed(E_FAIL);

    ComputeProgram* result = program.get();
    m_computePrograms.emplace(key, std::move(program));
    return result;
}

void ShaderProgramCache::RebuildAll()
{
    size_t failed = 0;
    for (auto& [key, program] : m_computePrograms)
        if (!program->Build())
            ++failed;

    if (failed > 0)
        spdlog::warn("Rebuilt {} of {} compute pipeline states ({} kept their previous PSO)",
                     m_computePrograms.size() - failed, m_computePrograms.size(), failed);
    else
        spdlog::info("Rebuilt {} compute pipeline states", m_computePrograms.size());
}
