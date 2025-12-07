#include "pch.h"
#include "StringId.h"

struct StorageView
{
	int offset;
	int size;
};

// TODO: possibly change Get functions to wrapping these in a struct/class with initialization on fist use
//  https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
static std::unordered_map<StringHash, StorageView>* g_stringHashToStorageViewLookup;
static std::vector<char>* g_sourceStringStorage;

static std::unordered_map<StringHash, StorageView>* GetLookup()
{
	if (g_stringHashToStorageViewLookup == nullptr)
	{
		g_stringHashToStorageViewLookup = new std::unordered_map<StringHash, StorageView>();
	}

	return g_stringHashToStorageViewLookup;
}

static std::vector<char>* GetStorage()
{
	if (g_sourceStringStorage == nullptr)
	{
		g_sourceStringStorage = new std::vector<char>();
	}

	return g_sourceStringStorage;
}

static bool IsRegistered(StringHash hash)
{
	return GetLookup()->find(hash) != GetLookup()->end();
}

static void RegisterMapping(StringHash hash, std::string_view str)
{
	assert(str.size() < INT_MAX);

	// Check this hash has not been already created before. 
	// If we did we have a collision, and the user needs to use a different string.
	if (IsRegistered(hash))
	{
		StorageView view = GetLookup()->operator[](hash);
		std::string_view collidedWithStr = std::string_view(GetStorage()->data() + view.offset, view.size);
		
		if (collidedWithStr.compare(str) == 0)
		{
			// This is a programmer error if it happens.
			SPDLOG_ERROR("Trying to register duplicate hash.String: \"{}\"", str.data());
			assert(false && "Trying to register duplicate hash. See console for details");
		}
		else
		{
			// This is bad luck if it happens, just choose a different string to hash.
			SPDLOG_ERROR("Hash collision for \"{}\" (collided with \"%{}\"). You need to use a different string.",
				 str.data(), collidedWithStr.data());
			// TODO : might wanna consider not crashing every time we rename a game object in the future...
			assert(false && "Hash collision! Use a different string. See console for details");
		}

		return;
	}

	const int sizeBeforeCopy = (int)GetStorage()->size();
	const int numberChars = (int)str.size();

	StorageView view{};
	view.offset = sizeBeforeCopy;
	view.size = numberChars;

	// copy the string to storage. Note : this might reallocate storage.
	for (size_t i = 0; i < numberChars; i++)
	{
		const char c = str.data()[i];
		GetStorage()->push_back(c);
	}

	GetLookup()->insert(std::make_pair(hash, view));
}

StringId StringId::New(const char* sz)
{
	return New(sz, const_strlen(sz));
}

StringId StringId::New(const char* s, size_t len)
{
	auto hash = StringHash(s, len);
	RegisterMapping(hash, std::string_view(s, len));

	return StringId(hash);
}

StringId StringId::New(std::string_view str)
{
	return New(str.data(), str.size());
}

StringId StringId::Existing(const char* sz)
{
	return Existing(sz, const_strlen(sz));
}

StringId StringId::Existing(const char* s, size_t len)
{
	auto hash = StringHash(s, len);

	if (IsRegistered(hash) == false)
	{
		SPDLOG_ERROR("Tried to get StringId that does not exist. string: {}", s);
		assert(false && "Tried to get StringId that does not exist");
	}

	return StringId(hash);
}

StringId StringId::Existing(std::string_view str)
{
	return Existing(str.data(), str.size());
}

bool StringId::TryGetExisting(const char* sz, StringId& outStringId)
{
	return TryGetExisting(sz, const_strlen(sz), outStringId);
}

bool StringId::TryGetExisting(const char* sz, size_t len, StringId& outStringId)
{
	auto hash = StringHash(sz, len);

	if (IsRegistered(hash))
	{
		outStringId = StringId(hash);
		return true;
	}

	return false;
}

bool StringId::TryGetExisting(std::string_view str, StringId& outStringId)
{
	return TryGetExisting(str.data(), str.size(), outStringId);
}

std::string_view StringId::GetUnderlyingString() const
{
	const auto [offset, size] = GetLookup()->operator[](m_hash);
	const char* basePtr = GetStorage()->data() + offset;
	const std::string_view strView(basePtr, size);
	return strView; // Note : this object is only valid until the underlying storage relocates. Do not store this. Copy this away if needed.
}

StringId::StringId(const char* sz) : StringId(sz, const_strlen(sz))
{
}

StringId::StringId(const char* sz, size_t len)
{
	m_hash = StringHash(sz, len);
}
StringId::StringId(std::string_view str) : StringId(str.data(), str.size())
{
}

bool StringId::Exists(StringId id)
{
	return IsRegistered(id);
}
