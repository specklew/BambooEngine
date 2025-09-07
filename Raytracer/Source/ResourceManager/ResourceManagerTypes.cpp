#include "pch.h"
#include "ResourceManager/ResourceManagerTypes.h"

static constexpr int MAX_PATH_LENGTH = 255;
static constexpr int SCRATCH_SIZE = MAX_PATH_LENGTH + 1; // +1 for null terminator
static char g_scratch[SCRATCH_SIZE] = {}; // 255 limit + null terminator

AssetId::AssetId(const char* szPath) : AssetId(szPath, strlen(szPath)){}

AssetId::AssetId(const char* path, const size_t len)
{
    assert(len < MAX_PATH_LENGTH && "Path is too long");

    // Copy the path into the scratch buffer.
    const errno_t ret = memcpy_s(g_scratch, MAX_PATH_LENGTH, path, len);
    assert(ret == 0 && "Failed to copy AssetId path into scratch buffer");
    g_scratch[len] = '\0';

    // Convert all characters to lowercase.
    // This is done because paths are case-insensitive on Windows (usually),
    // but our hash function is case-sensitive.
    for (size_t i = 0; i < len; i++)
    {
        g_scratch[i] = static_cast<char>(tolower(g_scratch[i]));
    }

    // convert all slashes to forward slashes
    for (size_t i = 0; i < len; i++)
    {
        if (g_scratch[i] == '\\')
        {
            g_scratch[i] = '/';
        }
    }

    // Store the resulting asset id
    m_path = std::string(g_scratch, len);
}

AssetId::AssetId(const std::string_view str)
    : AssetId(str.data(), str.size())
{
}

std::string_view AssetId::GetExtension() const
{
    const size_t lastSlash = m_path.find_last_of(".");
    if (lastSlash == std::string::npos)
    {
        assert(false && "Asset Id must always have extension");
        return {};
    }

    return std::string_view(m_path.c_str() + lastSlash); // includes the dot, to conform with std::filesystem::path::extension()
}

std::string_view AssetId::GetFileName() const
{
    const size_t lastSlash = m_path.find_last_of("/\\"); // catch both slashes
    if (lastSlash == std::string::npos)
    {
        return std::string_view(m_path);
    }

    return std::string_view(m_path.c_str() + lastSlash + 1);
}

bool AssetId::IsExtensionEqual(const char* extension) const
{
    return GetExtension().compare(extension) == 0;
}
