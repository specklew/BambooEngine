#pragma once

#include "ResourceManager/ResourceManagerTypes.h"

struct ShaderMetadata
{
    char szPathWithinResources[64]; // e.g. "Shaders/MyShader.hlsl"
    char szEntrypoint[32]; // e.g. "VS"
    char szTarget[32]; // e.g. "vs_6_0"
    // Optional, may be empty. Sized for a base sidecar's defines plus every
    // compile-time vendor lever's, since a variant concatenates the two.
    char szDefines[256]; // e.g. "-D MyDefine=1 -D another=2"

    std::filesystem::file_time_type lastCompilationTime = {}; // Remember to set this after compiling the shader. TODO : can we do this automatically?

    static ShaderMetadata Deserialize(const char* szSerializedData);
};

struct Shader
{
    ResourceId id;
    Microsoft::WRL::ComPtr<IDxcBlob> bytecode = nullptr;
    Microsoft::WRL::ComPtr<IDxcBlob> reflection = nullptr; // DXC reflection blob, null if DXC produced none
    ShaderMetadata metadata = {}; // TODO : is this only stored for recompilation? if so, maybe editor only?

    explicit Shader(ResourceId id) : id(id) {}
    static void ReleaseResources() {}
};
