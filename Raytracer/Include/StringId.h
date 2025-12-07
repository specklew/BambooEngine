#pragma once

#include "StringHash.h"

// TODO : We could have this decay to regular string hash in release mode, when the debug strings are not necessary (not so sure, why would we?)
// TODO : visual studio debugger helper? https://learn.microsoft.com/en-us/visualstudio/debugger/viewing-data-in-the-debugger?view=vs-2022

/// <summary>
/// 
/// Hashed String Id that allows to retrieve the underlying string for debug purposes.
/// The underlying string should not be used for comparison, mapping, etc. 
/// This class itself should, as it only stores the 32 bit hash of the string.
/// Hash is guaranteed to be unique. Any collisions will be made known and the user
/// will have to change the string they are trying to hash.
/// 
/// The user is required to follow New/Existing semantics.
/// 
/// Example usage:
/// <code>
/// GameObject* go = CreateGameObject(StringId::New("skySphere")); // Note : creation of new id registers it
/// go->primitiveHandle = ResourceManager::Get().primitives.Get(StringId::Existing("cube")); // Note : retrieval of exising id that had been created earlier
/// </code>
/// 
/// </summary>
struct StringId
{
	// Create and register a new StringId
	static StringId New(const char* sz);
	static StringId New(const char* sz, size_t len);
	static StringId New(std::string_view str);

	// Return an existing string id.
	static StringId Existing(const char* sz);
	static StringId Existing(const char* sz, size_t len);
	static StringId Existing(std::string_view str);
	
	// Outputs the StringId via the outStringId, if it exists. Returns whether it exists.
	static bool TryGetExisting(const char* sz, StringId& outStringId);
	static bool TryGetExisting(const char* sz, size_t len, StringId& outStringId);
	static bool TryGetExisting(std::string_view str, StringId& outStringId);

	// Explicitly allow copying, since this should usually be passed by value since its only 4 bytes
	constexpr StringId(const StringId& other) = default;  

	// Return a temporary string_view to the string that the StringId was created from. 
	// The string_view is only valid until the underlying storage relocates. Do not store it.
	std::string_view GetUnderlyingString() const;

	bool IsValid() const { return m_hash.IsValid(); }
	constexpr operator StringHash() const { return m_hash; }

	explicit StringId(StringHash hash) : m_hash(hash) {} // TODO : construction cleanup
	explicit StringId() : m_hash(StringHash()) {}
	explicit StringId(const char* sz);
	explicit StringId(const char* sz, size_t len);
	explicit StringId(std::string_view str);

private:
	static bool Exists(StringId id);
	StringHash m_hash;

	friend class StringIdHelper; // remove the helper
};

class StringIdHelper
{
	// Expose this 2 functions to resource manager

	static StringId InvalidStringId() { return StringId(StringHash()); }
	static bool Exists(StringId id) { return StringId::Exists(id); }
	friend class ResourceManager;
};

// Allow to be used as a key to STL maps
template<>
struct std::hash<StringId>
{
	constexpr std::size_t operator()(const StringId& id) const noexcept
	{
		const StringHash h = static_cast<StringHash>(id);
		constexpr std::hash<StringHash> hash;
		return hash.operator()(h);
	}
};

constexpr bool operator==(const StringId& lhs, const StringId& rhs)
{
	return static_cast<StringHash>(lhs).computedHash == static_cast<StringHash>(rhs).computedHash;
}

constexpr bool operator!=(const StringId& lhs, const StringId& rhs)
{
	return static_cast<StringHash>(lhs).computedHash != static_cast<StringHash>(rhs).computedHash;
}
