#pragma once
#include "Random.h"

static constexpr size_t const_strlen(const char* s)
{
    size_t size = 0;
    while (s[size]) { size++; }
    return size;
}

struct StringHash
{
    uint32_t computedHash;

    constexpr StringHash() noexcept : computedHash(0) {}
	
    constexpr explicit StringHash(uint32_t hash) noexcept : computedHash(hash) {}

    constexpr explicit StringHash(const char* s) noexcept : computedHash(0)
    {
        computedHash = RaytracerRandom::fnv1a_32(s, const_strlen(s));
    }
	
    constexpr explicit StringHash(const char* s, size_t count) noexcept : computedHash(0)
    {
        computedHash = RaytracerRandom::fnv1a_32(s, count);
    }
	
    constexpr explicit StringHash(std::string_view s) noexcept : computedHash(0)
    {
        computedHash = RaytracerRandom::fnv1a_32(s.data(), s.size());
    }

    StringHash(const StringHash& other) = default;
    constexpr operator uint32_t() const noexcept { return computedHash; }
    [[nodiscard]] bool IsValid() const { return computedHash != 0; }
};

// Allow this to be a key to std::unordered_map 
// https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key_
// https://en.cppreference.com/w/cpp/utility/hash

// Custom specialization of std::hash can be injected in namespace std.
template<>
struct std::hash<StringHash>
{
    constexpr std::size_t operator()(const StringHash& s) const noexcept
    {
        return s.computedHash;
    }
};

constexpr bool operator==(const StringHash& lhs, const StringHash& rhs)
{
    return lhs.computedHash == rhs.computedHash;
}

constexpr bool operator!=(const StringHash& lhs, const StringHash& rhs)
{
    return lhs.computedHash != rhs.computedHash;
}
