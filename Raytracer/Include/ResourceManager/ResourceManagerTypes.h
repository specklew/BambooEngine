#pragma once
#include <filesystem>

#include "Resources/StringId.h"

template<typename IndexType>
struct ResourceHandle
{
	IndexType index;

	constexpr ResourceHandle() : index(0) {}
	constexpr explicit ResourceHandle(uint32_t index) : index(index) {}

	[[nodiscard]] constexpr bool IsValid() const { return index != Invalid().index; }
	
	constexpr static ResourceHandle Invalid() { return ResourceHandle(-1); }

	friend class ResourceManager;
};

typedef ResourceHandle<uint16_t> TextureHandle;
typedef ResourceHandle<uint16_t> MaterialHandle;
typedef ResourceHandle<uint8_t> MeshHandle;
typedef ResourceHandle<uint8_t> PrimitiveHandle;
typedef ResourceHandle<uint8_t> SpecialMaterialHandle;
typedef ResourceHandle<uint8_t> ShaderHandle;
typedef ResourceHandle<uint8_t> FontHandle;
typedef ResourceHandle<uint8_t> AnimationHandle;

template<typename IndexType>
struct std::hash<ResourceHandle<IndexType>>
{
	constexpr std::size_t operator()(const ResourceHandle<IndexType>& s) const noexcept
	{
		return std::hash<int>()(s.index);
	}
};

template<typename IndexType>
constexpr inline bool operator==(const ResourceHandle<IndexType>& lhs, const ResourceHandle<IndexType>& rhs)
{
	return lhs.index == rhs.index;
}

template<typename IndexType>
constexpr inline bool operator!=(const ResourceHandle<IndexType>& lhs, const ResourceHandle<IndexType>& rhs)
{
	return lhs.index != rhs.index;
}

// TODO : resource id fuss - maybe just have a regular constuctor that always registers and there's no assertions and new and existing?
// Because it makes the API difficult to use, and I feel like we jump through hoops and still its the resource manager
// that should have the asserts, not the resource id itself.
typedef StringId ResourceId; // Do not use file extension in IDs (e.g. pass "Textures/MyTexture" instead of "Textures/MyTexture.png")

class AssetId
{
public:
	AssetId() = default;
	explicit AssetId(const char* szPath);
	explicit AssetId(const char* path, size_t len);
	explicit AssetId(std::string_view str);

	[[nodiscard]] std::string_view GetExtension() const; // will return the dot with extension, e.g. ".dds"
	[[nodiscard]] std::string_view GetFileName() const; // with extension; e.g. "Resources/textures/MyTexture.dds" will give "MyTexture.dds"
	[[nodiscard]] bool IsExtensionEqual(const char* extension) const; // Pass dot with extension, e.g. ".dds"

	[[nodiscard]] bool IsValid() const { return !m_path.empty(); }
	[[nodiscard]] std::filesystem::path AsPath() const { return std::filesystem::path(m_path); } // Note: copy
	[[nodiscard]] std::string AsString() const { return std::string(m_path); } // Note: copy

	[[nodiscard]] const char* c_str() const { return m_path.c_str(); } // Note: pointer to internal buffer
	[[nodiscard]] size_t size() const { return m_path.size(); }

private:
	std::string m_path;
};

#define ASSERT_VALID_RESOURCE_ID(x) assert((x).IsValid())
#define ASSERT_VALID_ASSET_ID(x) assert((x).IsValid())

template<typename T>
struct TypedBufferView
{
	const T* data;
	size_t numElements;

	TypedBufferView() : data(nullptr), numElements(0) {}
	TypedBufferView(const T* data, size_t numElements) : data(data), numElements(numElements) { assert(data); }
};
