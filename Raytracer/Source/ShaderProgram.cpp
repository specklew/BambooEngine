#include "pch.h"
#include "ShaderProgram.h"

#include "ResourceManager/ResourceManager.h"
#include "Shader.h"
#include "Utils/Utils.h"

void ComputeProgram::Build()
{
    auto& resourceManager = ResourceManager::Get();
    auto  shaderHandle    = resourceManager.GetOrLoadShader(AssetId(m_shaderAssetPath));
    auto  bytecode        = resourceManager.shaders.GetResource(shaderHandle).bytecode;

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = m_rootSignature.Get();
    desc.CS             = CD3DX12_SHADER_BYTECODE(bytecode->GetBufferPointer(), bytecode->GetBufferSize());

    ThrowIfFailed(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_pipelineState)));
    m_pipelineState->SetName(m_debugName.c_str());
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
    program->Build();

    ComputeProgram* result = program.get();
    m_computePrograms.emplace(key, std::move(program));
    return result;
}

void ShaderProgramCache::RebuildAll()
{
    for (auto& [key, program] : m_computePrograms)
        program->Build();

    spdlog::info("Rebuilt {} compute pipeline states", m_computePrograms.size());
}
