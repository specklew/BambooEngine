#include <pch.h>
#include "Shader.h"
#include "rapidjson/document.h"

// ReSharper disable CppClangTidyReadabilityMisplacedArrayIndex

ShaderMetadata ShaderMetadata::Deserialize(const char* szSerializedData)
{
    using namespace rapidjson;
    Document doc;
    doc.Parse(szSerializedData);
    assert(doc.HasParseError() == false);

    ShaderMetadata data{};
    strcpy_s(data.szPathWithinResources, doc["pathWithinResources"].GetString());
    strcpy_s(data.szEntrypoint, doc["entrypoint"].GetString());
    strcpy_s(data.szTarget, doc["target"].GetString());
    if (doc.HasMember("defines"))
    {
        strcpy_s(data.szDefines, doc["defines"].GetString());
    }

    return data;
}
