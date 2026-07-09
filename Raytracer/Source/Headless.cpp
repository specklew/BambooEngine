#include "pch.h"
#include "Headless.h"

#include "rapidjson/document.h"

#include <fstream>
#include <sstream>

namespace
{
    std::string WideToUtf8(const std::wstring& w)
    {
        if (w.empty()) return {};
        const int size = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
        std::string out(static_cast<size_t>(size), '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), size, nullptr, nullptr);
        return out;
    }

    std::vector<std::string> SplitCsv(const std::string& value)
    {
        std::vector<std::string> out;
        std::stringstream ss(value);
        std::string item;
        while (std::getline(ss, item, ','))
            if (!item.empty())
                out.push_back(item);
        return out;
    }
}

HeadlessArgs ParseHeadlessArgs(int argc, wchar_t** argv)
{
    HeadlessArgs args;

    auto valueOf = [&](int& i) -> std::string {
        return (i + 1 < argc) ? WideToUtf8(argv[++i]) : std::string{};
    };

    for (int i = 0; i < argc; ++i)
    {
        const std::string flag = WideToUtf8(argv[i]);
        if (flag == "--headless")        args.headless = true;
        else if (flag == "--scene")      args.scene = valueOf(i);
        else if (flag == "--places")     args.places = SplitCsv(valueOf(i));
        else if (flag == "--techniques") args.techniques = SplitCsv(valueOf(i));
        else if (flag == "--seconds")    args.seconds = std::stof(valueOf(i));
        else if (flag == "--out")        args.outDir = valueOf(i);
    }

    return args;
}

HeadlessConfig LoadHeadlessConfig(const std::string& path)
{
    HeadlessConfig config;

    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        spdlog::info("Headless config not found at {}, using built-in defaults", path);
        return config;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    const std::string json = ss.str();

    rapidjson::Document doc;
    if (doc.Parse(json.c_str()).HasParseError() || !doc.IsObject())
    {
        spdlog::warn("Headless config at {} is not valid JSON, using built-in defaults", path);
        return config;
    }

    auto readUint  = [&](const char* key, uint32_t& out) { if (doc.HasMember(key) && doc[key].IsUint())   out = doc[key].GetUint(); };
    auto readFloat = [&](const char* key, float& out)    { if (doc.HasMember(key) && doc[key].IsNumber()) out = doc[key].GetFloat(); };
    auto readBool  = [&](const char* key, bool& out)     { if (doc.HasMember(key) && doc[key].IsBool())   out = doc[key].GetBool(); };

    readUint("width",  config.width);
    readUint("height", config.height);
    readUint("spp",     config.spp);
    readUint("bounces", config.bounces);
    readBool ("postProcessEnabled", config.postProcessEnabled);
    readFloat("exposure",   config.exposure);
    readFloat("contrast",   config.contrast);
    readFloat("saturation", config.saturation);
    readFloat("lift",       config.lift);
    readFloat("indirectSkyClamp", config.indirectSkyClamp);
    readBool ("skyLighting", config.skyLighting);
    readFloat("defaultSeconds", config.defaultSeconds);
    if (doc.HasMember("outputDir") && doc["outputDir"].IsString())
        config.outputDir = doc["outputDir"].GetString();

    if (doc.HasMember("lights") && doc["lights"].IsArray())
    {
        auto readVec3 = [](const rapidjson::Value& entry, const char* key, float out[3]) {
            if (entry.HasMember(key) && entry[key].IsArray() && entry[key].Size() == 3)
                for (rapidjson::SizeType i = 0; i < 3; ++i)
                    if (entry[key][i].IsNumber())
                        out[i] = entry[key][i].GetFloat();
        };

        for (const auto& entry : doc["lights"].GetArray())
        {
            if (!entry.IsObject())
                continue;

            HeadlessLight light;
            if (entry.HasMember("type") && entry["type"].IsString())
                light.type = entry["type"].GetString();
            readVec3(entry, "position",  light.position);
            readVec3(entry, "direction", light.direction);
            readVec3(entry, "color",     light.color);
            if (entry.HasMember("intensity") && entry["intensity"].IsNumber()) light.intensity = entry["intensity"].GetFloat();
            if (entry.HasMember("range")     && entry["range"].IsNumber())     light.range     = entry["range"].GetFloat();
            config.lights.push_back(light);
        }
    }

    return config;
}
