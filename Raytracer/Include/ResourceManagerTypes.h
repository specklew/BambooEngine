#pragma once
#include <filesystem>

#include "StringId.h"

template<int TypeIndex, typename TIndex>
struct ResourceHandle
{
	TIndex index;

	constexpr ResourceHandle() : index(0) {}
	constexpr explicit ResourceHandle(uint32_t index) : index(index) {}
	constexpr inline bool IsValid() const { return index != Invalid().index; }
	constexpr inline static ResourceHandle Invalid() { return ResourceHandle(-1); }

	friend class ResourceManager;
};

typedef ResourceHandle<0, uint16_t> TextureHandle;
typedef ResourceHandle<1, uint16_t> MaterialHandle;
typedef ResourceHandle<2, uint8_t> MeshHandle;
typedef ResourceHandle<3, uint8_t> PrimitiveHandle;
typedef ResourceHandle<4, uint8_t> SpecialMaterialHandle;
typedef ResourceHandle<5, uint8_t> ShaderHandle;
typedef ResourceHandle<6, uint8_t> FontHandle;
typedef ResourceHandle<7, uint8_t> AnimationHandle;

template<int TypeIndex, typename THandle>
struct std::hash<ResourceHandle<TypeIndex, THandle>>
{
	constexpr std::size_t operator()(const ResourceHandle<TypeIndex, THandle>& s) const noexcept
	{
		return std::hash<int>()(s.index);
	}
};

template<int TypeIndex, typename THandle>
constexpr inline bool operator==(const ResourceHandle<TypeIndex, THandle>& lhs, const ResourceHandle<TypeIndex, THandle>& rhs)
{
	return lhs.index == rhs.index;
}

template<int TypeIndex, typename THandle>
constexpr inline bool operator!=(const ResourceHandle<TypeIndex, THandle>& lhs, const ResourceHandle<TypeIndex, THandle>& rhs)
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
