#pragma once
#include "Helpers.h"
#include "ResourceManagerTypes.h"

struct Shader;

template <typename T, typename THandle, int MaxAmount>
class ResourceStorage
{
public:
    [[nodiscard]] THandle Get(const ResourceId id) const
    {
        ASSERT_VALID_RESOURCE_ID(id);

        if (ContainsResource(id))
        {
            return m_resourceIdToHandleLookup.at(id);
        }
        else
        {
            const std::string_view str = id.GetUnderlyingString();
            SPDLOG_WARN("Tried to get a resource that wasn't loaded. ResourceId: ", str.data());
            assert(false && "Tried to get a resource that wasn't loaded. See console for details.");
            return THandle::Invalid();
        }
    }

    [[nodiscard]] ResourceId GetResourceId(const THandle handle) const
    {
        const T& resource = GetResource(handle);
        return resource.id;
    }

    THandle Put(T&& resource) // TODO : address this rider warning, I think it is moved via std::move(), no?
    {
        ASSERT_VALID_RESOURCE_ID(resource.id);
        assert(ContainsResource(resource.id) == false && "The resource is already in the storage");
        assert(m_stackTop < MaxAmount && "Out of resource stack space");

        // TODO : address this
        // If we change this to no longer append to the end or somehow shift indices,
        // texture indexing in the shader will break. See GetTextureDescriptorIndex() for more info.
        ResourceId id = resource.id;
        new (&m_data[m_stackTop]) T(std::move(resource)); // unordered_map have some issues with move assignment operator,
                                                          // so I needed to change it to move constructor,
                                                          // but this is not allocation, but constructor call in existing memory
        auto rh = THandle(m_stackTop);
        m_stackTop++;
        m_resourceIdToHandleLookup[id] = rh;

        return rh;
    }

    [[nodiscard]] T& GetResource(const THandle handle)
    {
        assert(handle.IsValid() && handle.index < m_stackTop);
        return m_data[handle.index];
    }

    [[nodiscard]] T& GetResource(const ResourceId id)
    {
        const THandle handle = Get(id);
        assert(handle.IsValid() && handle.index < m_stackTop);
        return m_data[handle.index];
    }

    [[nodiscard]] const T& GetResource(const THandle handle) const
    {
        assert(handle.IsValid() && handle.index < m_stackTop);
        return m_data[handle.index];
    }

    __forceinline bool ContainsResource(const ResourceId id) const
    {
        return m_resourceIdToHandleLookup.find(id) != m_resourceIdToHandleLookup.end();
    }

    [[nodiscard]] TypedBufferView<T> GetBufferView() const
    {
        return TypedBufferView<T>(m_data, m_stackTop);
    }

    void ForEach(std::function<void(T&)> func) // TODO: Maybe support built-in foreach by implement begin() and end() methods?
    {
        for (uint32_t i = 0; i < m_stackTop; i++)
        {
            func(m_data[i]);
        }
    }

private:

    void Allocate()
    {
        assert(m_data == nullptr);
        assert(m_resourceIdToHandleLookup.size() == 0);

        const size_t numBytes = MaxAmount * sizeof(T);
        void* ptr = malloc(numBytes);
        assert(ptr && "Failed to allocate");
        memset(ptr, 0, numBytes);
        m_data = (T*)ptr;

    }

    void Release()
    {
        for (uint32_t i = 0; i < m_stackTop; i++)
        {
            m_data[i].ReleaseResources();
        }
        
        AssertFreeClear((void**)&m_data);
    }

    __forceinline bool IsInitialized() const
    {
        return m_data != nullptr;
    }

    T* m_data = nullptr;
    uint32_t m_stackTop = 0;
    std::unordered_map<ResourceId, THandle> m_resourceIdToHandleLookup = {};

    friend class ResourceManager;
};

class ResourceManager
{
    // TODO : add support for unloading resources
    // TODO : remove the need for the renderer to be passed around
    // TODO : address isSkybox hack

    // Implementation notes:
    // - Handle indices are used to index GPU buffers.

public:
    static constexpr int NUM_SPECIAL_MATERIALS = 32;
    static constexpr int NUM_MESHES = 256;
    static constexpr int NUM_ANIMATIONS = 256;
    static constexpr int NUM_PRIMITIVES = 256;
    static constexpr int NUM_SKELETAL_PRIMITIVES = 256;
    static constexpr int NUM_SHADERS = 32;
    static constexpr int NUM_FONTS = 4;


public:
    static ResourceManager& Get();

    void AllocateResources();
    void ReleaseResources();

    [[nodiscard]] ResourceId CreateNewResourceId(const AssetId& assetId) const;
    [[nodiscard]] ResourceId GetExistingResourceId(const AssetId& assetId) const;

    // TODO : ResourceStorageExtended? probably sort out parameters first? this would reduce code duplication
    [[nodiscard]] ShaderHandle GetOrLoadShader(const AssetId& assetId);
    
    [[nodiscard]] const AssetId& GetAssetId(ResourceId id) const;

    bool RecompileAllShaders(); // return true if everything was recompiled successfully
    bool RecompileOutdatedShadersIfAny(); // return true if something was recompiled

    ResourceStorage<Shader, ShaderHandle, NUM_SHADERS> shaders;

private:
    bool TryGetExistingResourceId(const AssetId& assetId, ResourceId& outResourceId) const;

    [[nodiscard]] ShaderHandle LoadShader(const AssetId& assetId);

    __forceinline void AssertExistsInAssetIdLookup(ResourceId id) const; // inline won't work unless defined in header, right?

    std::unordered_map<ResourceId, AssetId> m_assetIdLookup; // TODO : I don't like having asset ids scattered around the heap. Could we use a string store like for StringId?
};
