#pragma once

#include "ResourceManager/ResourceManagerTypes.h"

struct ShaderMetadata
{
    char szPathWithinResources[64]; // e.g. "Shaders/MyShader.hlsl"
    char szEntrypoint[16]; // e.g. "VS"
    char szTarget[16]; // e.g. "vs_6_0"
    // Optional, may be nullptr
    char szDefines[64]; // e.g. "-D MyDefine=1 -D another=2"

    std::filesystem::file_time_type lastCompilationTime = {}; // Remember to set this after compiling the shader. TODO : can we do this automatically?

    static ShaderMetadata Deserialize(const char* szSerializedData);
};

struct Shader
{
    ResourceId id;
    Microsoft::WRL::ComPtr<IDxcBlob> bytecode = nullptr;
    ShaderMetadata metadata = {}; // TODO : is this only stored for recompilation? if so, maybe editor only?

    explicit Shader(ResourceId id) : id(id) {}
    static void ReleaseResources() {}
};
