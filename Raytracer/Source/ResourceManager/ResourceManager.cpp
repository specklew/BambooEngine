#include "pch.h"
#include "ResourceManager/ResourceManager.h"

#include "Utils/Utils.h"
#include "Shader.h"
#include "ShaderCompilation.h"
#include "VendorLevers.h"

ResourceManager& ResourceManager::Get()
{
    static ResourceManager rm;
    if (rm.shaders.IsInitialized() == false)
    {
        rm.AllocateResources();
    }
    return rm;
}

void ResourceManager::AllocateResources()
{
    shaders.Allocate();
}

void ResourceManager::ReleaseResources()
{
    shaders.Release();
}

// TODO : can we do this differently?
static std::string_view GetStringViewForResourceId(const std::string& narrowString, const AssetId& assetId)
{
    // ResourceId is path without extension. Only narrow strings are supported.

    const char* str = narrowString.c_str();
    const int assetIdLength = (int)narrowString.length();
    const int extensionLength = (int)assetId.GetExtension().length();

    return std::string_view(str, assetIdLength - extensionLength);
}

ResourceId ResourceManager::GetResourceId(const AssetId& assetId)
{
    const std::string narrowString = assetId.AsString();
    const std::string_view strView = GetStringViewForResourceId(narrowString, assetId);
    return ResourceId::New(strView);
}

ResourceId ResourceManager::CreateNewResourceId(const AssetId& assetId) const
{
    const std::string narrowString = assetId.AsString();
    const std::string_view strView = GetStringViewForResourceId(narrowString, assetId);
    return ResourceId::New(strView);
}

ResourceId ResourceManager::GetExistingResourceId(const AssetId& assetId) const
{
    const std::string narrowString = assetId.AsString();
    const std::string_view strView = GetStringViewForResourceId(narrowString, assetId);
    return ResourceId::Existing(strView);
}

bool ResourceManager::TryGetExistingResourceId(const AssetId& assetId, ResourceId& outResourceId) const
{
    const std::string narrowString = assetId.AsString();
    const std::string_view strView = GetStringViewForResourceId(narrowString, assetId);
    bool exists = ResourceId::TryGetExisting(strView, outResourceId);
    return exists;
}

// TODO : get buffer as param
// Note : Allocates memory; caller takes ownership of the returned pointer.
static const char* LoadAssetFile(const AssetId& id)
{
    ASSERT_VALID_ASSET_ID(id);

    constexpr int bufferByteSize = 2048 * 4; // 8KiB buffer, TODO : use some pre-allocated scratch memory for loading asset files.
    char* buffer = new char[bufferByteSize];

    ReadTextFromFile(id.c_str(), buffer, bufferByteSize);

    return buffer;
}

ShaderHandle ResourceManager::GetOrLoadShader(const AssetId& assetId)
{
    ASSERT_VALID_ASSET_ID(assetId);

    ResourceId id = StringIdHelper::InvalidStringId();
    bool idExists = TryGetExistingResourceId(assetId, id);
    if (idExists)
    {
        AssertExistsInAssetIdLookup(id);
        return shaders.Get(id);
    }
    else
    {
        assert(shaders.ContainsResource(id) == false); // if the id doesn't exist then the resource must not be loaded.
        return LoadShader(assetId);
    }
}

ShaderHandle ResourceManager::LoadShader(const AssetId& assetId)
{
    ASSERT_VALID_ASSET_ID(assetId);

    // A variant asset id is "<base .shader path>|<lever key>". The base file still
    // supplies source, entry point and target; the key names levers whose defines
    // are appended. Variants are ordinary entries in this table, so the hot-reload
    // walk below already covers every one of them — and the key carries lever
    // NAMES rather than raw defines because AssetId lowercases what it is given,
    // and DXC define names are case-sensitive.
    const std::string full = assetId.AsString();
    const size_t      bar  = full.find('|');

    std::string baseString = full;
    std::string variantKey;
    if (bar != std::string::npos)
    {
        // "path/x.rg|onesample.shader" -> base "path/x.rg.shader", key "onesample".
        const size_t dot = full.find('.', bar);
        variantKey = full.substr(bar + 1, dot == std::string::npos ? std::string::npos : dot - bar - 1);
        baseString = full.substr(0, bar) + (dot == std::string::npos ? std::string{} : full.substr(dot));
    }
    const AssetId baseAsset(baseString);

    const char* buffer = LoadAssetFile(baseAsset);
    ShaderMetadata meta = ShaderMetadata::Deserialize(buffer);
    delete[] buffer;

    if (!variantKey.empty())
    {
        const std::string extra = VendorLevers::DefinesForKey(variantKey);
        std::string combined = meta.szDefines;
        if (!combined.empty() && !extra.empty())
            combined += ' ';
        combined += extra;
        if (combined.size() >= sizeof(meta.szDefines))
            spdlog::error("Shader variant defines too long for '{}': {}", full, combined);
        else
            strcpy_s(meta.szDefines, combined.c_str());
        spdlog::info("Compiling shader variant {} ({})", full, meta.szDefines);
    }

    Shader shader(CreateNewResourceId(assetId));
    shader.bytecode = CompileShader(meta, &shader.reflection);
    shader.metadata = meta;
    shader.metadata.lastCompilationTime = std::filesystem::_File_time_clock::now();

    assert(m_assetIdLookup.find(shader.id) == m_assetIdLookup.end() && "AssetId is already stored");
    m_assetIdLookup[shader.id] = assetId; // store a copy of the asset id for lookup

    return shaders.Put(std::move(shader));
}

void ResourceManager::AssertExistsInAssetIdLookup(ResourceId id) const
{
    assert(m_assetIdLookup.find(id) != m_assetIdLookup.end() && "This resource should exist in asset id lookup");
}

const AssetId& ResourceManager::GetAssetId(ResourceId id) const
{
    if (m_assetIdLookup.find(id) == m_assetIdLookup.end())
    {
        std::string_view str = id.GetUnderlyingString();
        SPDLOG_INFO("The following ResourceId exists, but is not registered with AssetId Lookup."
            " This is not allowed. ResourceId: \"{}\"", str.data());
        assert(false && "Inconsistent AssetId Lookup. See console for details.");
    }

    return m_assetIdLookup.at(id);
}


bool ResourceManager::RecompileAllShaders()
{
    for (uint32_t i = 0; i < shaders.m_stackTop; i++)
    {
        Shader& shader = shaders.GetResource(ShaderHandle(i));
        Microsoft::WRL::ComPtr<IDxcBlob> newReflection;
        Microsoft::WRL::ComPtr<IDxcBlob> newBytecode = CompileShader(shader.metadata, &newReflection);
        if (newBytecode == nullptr)
        {
            return false;
        }
        shader.bytecode   = newBytecode;
        shader.reflection = newReflection;
        shader.metadata.lastCompilationTime = std::filesystem::_File_time_clock::now();
    }

    return true;
}

bool ResourceManager::RecompileOutdatedShadersIfAny()
{
    using namespace std::filesystem;

    bool anyShaderRecompiled = false;

    for (uint32_t i = 0; i < shaders.m_stackTop; i++)
    {
        Shader& shader = shaders.GetResource(ShaderHandle(i));

        path sourceFilePath = path("Resources/").concat(shader.metadata.szPathWithinResources);

        const auto lastWriteTime = last_write_time(sourceFilePath);

        if (lastWriteTime > shader.metadata.lastCompilationTime)
        {
            SPDLOG_INFO("Recompiling {}", shader.id.GetUnderlyingString());
            Microsoft::WRL::ComPtr<IDxcBlob>       newReflection;
            const Microsoft::WRL::ComPtr<IDxcBlob> newBytecode = CompileShader(shader.metadata, &newReflection);

            if (newBytecode != nullptr)
            {
                shader.bytecode   = newBytecode;
                shader.reflection = newReflection;
                shader.metadata.lastCompilationTime = _File_time_clock::now();
                anyShaderRecompiled = true;
            }
            else
            {
                SPDLOG_WARN("Shader recompilation failed! ({})", shader.metadata.szPathWithinResources);
            }
        }
    }

    return anyShaderRecompiled;
}
