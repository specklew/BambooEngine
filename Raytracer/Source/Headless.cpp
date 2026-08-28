#include "pch.h"
#include "Headless.h"

#include "Utils/CVars.h"
#include "VendorLevers.h"
#include "rapidjson/document.h"

#include <cstdlib>
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
        {
            // Underscores stand in for spaces so state/technique names survive
            // launchers that cannot forward quoted arguments (pixtool).
            std::replace(item.begin(), item.end(), '_', ' ');
            if (!item.empty())
                out.push_back(item);
        }
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
        else if (flag == "--states")     args.states = SplitCsv(valueOf(i));
        else if (flag == "--states-key") args.statesKey = valueOf(i);
        else if (flag == "--techniques") args.techniques = SplitCsv(valueOf(i));
        else if (flag == "--seconds")    args.seconds = std::stof(valueOf(i));
        else if (flag == "--budget")
        {
            const std::string budget = valueOf(i);
            const size_t      colon  = budget.find(':');
            const std::string kind   = budget.substr(0, colon);
            const std::string value  = colon == std::string::npos ? std::string{} : budget.substr(colon + 1);
            if (kind == "frames")       args.budgetFrames = static_cast<uint32_t>(std::stoul(value));
            else if (kind == "seconds") args.seconds      = std::stof(value);
            else
            {
                // Same class as an unparseable config: a rejected budget used to leave the
                // built-in default in place, so the run measured a budget nobody asked for
                // and still exited 0 with images on disk.
                spdlog::error("--budget expects frames:N or seconds:T, got '{}' — refusing to fall "
                              "back to the default budget, because that is a different measurement", budget);
                std::exit(2);
            }
        }
        else if (flag == "--images")      args.images = static_cast<uint32_t>(std::stoul(valueOf(i)));
        else if (flag == "--checkpoints") args.checkpoints = valueOf(i);
        else if (flag == "--warmup")      args.warmupSeconds = std::stof(valueOf(i));
        else if (flag == "--out")        args.outDir = valueOf(i);
        else if (flag == "--debug-views") args.debugViews = SplitCsv(valueOf(i));
        else if (flag == "--debug-layer") args.debugLayer = true;
        else if (flag == "--rdg-dump")    args.rdgDump = true;
        else if (flag == "--rdg-timings") args.rdgTimings = true;
        else if (flag == "--cvar")        args.cvarAssignments.push_back(valueOf(i));
        else if (flag == "--cvar-matrix") args.cvarMatrix = valueOf(i);
        else if (flag == "--config")      args.configPath = valueOf(i);
        else if (flag == "--levers")
        {
            // "--levers none" is how a matrix row asks for the baseline: an empty
            // value would be indistinguishable from the flag being absent.
            const std::string value = valueOf(i);
            args.leversSpecified = true;
            if (value != "none")
                args.levers = SplitCsv(value);
        }
    }

    return args;
}

// --cvar name=value and --levers a,b, applied together because both write CVars
// and both have to be able to beat the headless config, which is read later.
// Float or int is decided by which one the CVar system already knows the name as,
// so nothing has to state a type on the command line; an unknown name is a typo
// and says so rather than being ignored.
void ApplyCVarAssignment(const std::string& name, const std::string& value)
{
    const StringId id(name.c_str());

    if (CVarSystem::Get()->GetFloatCVar(id))
        CVarSystem::Get()->SetCVarFloat(id, std::stof(value));
    else if (CVarSystem::Get()->GetIntCVar(id))
        CVarSystem::Get()->SetCVarInt(id, std::stoi(value));
    else
    {
        spdlog::error("CVar '{}' is not a float or int CVar", name);
        return;
    }
    spdlog::info("cvar {} = {}", name, value);
}

void ApplyCommandLineOverrides(const HeadlessArgs& args)
{
    for (const std::string& assignment : args.cvarAssignments)
    {
        const size_t separator = assignment.find('=');
        if (separator == std::string::npos)
        {
            spdlog::error("--cvar expects name=value, got '{}'", assignment);
            continue;
        }
        ApplyCVarAssignment(assignment.substr(0, separator), assignment.substr(separator + 1));
    }

    if (!args.leversSpecified)
        return;

    // --levers states the WHOLE set: naming one lever turns the others off, so a
    // matrix row cannot inherit a leftover from the row before it.
    for (const VendorLever& lever : VendorLevers::Get().All())
        VendorLevers::Get().SetEnabled(lever.name, false);
    for (const std::string& name : args.levers)
        VendorLevers::Get().SetEnabled(name, true);
    spdlog::info("--levers {}", VendorLevers::Get().ActiveNames());
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
    std::string json = ss.str();

    // PowerShell's Set-Content and Out-File write a UTF-8 BOM by default, and
    // rapidjson treats it as a parse error. These files get edited from a shell as
    // often as from an editor, so skip it rather than reject the file.
    if (json.size() >= 3 && static_cast<unsigned char>(json[0]) == 0xEF
                         && static_cast<unsigned char>(json[1]) == 0xBB
                         && static_cast<unsigned char>(json[2]) == 0xBF)
        json.erase(0, 3);

    rapidjson::Document doc;
    if (doc.Parse(json.c_str()).HasParseError() || !doc.IsObject())
    {
        // Fatal, not a fallback. The built-in defaults are a DIFFERENT measurement
        // (720p, exposure 1.0, sky lighting on, the scene's own lights), so a run
        // that quietly swapped them in would report conditions it never rendered
        // under. The old behaviour warned and continued, which is the same failure
        // with a line in the log.
        spdlog::error("Headless config at {} is not valid JSON — refusing to fall back to built-in "
                      "defaults, because those are a different measurement", path);
        std::exit(2);
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
    readBool ("injectionReuseInMis", config.injectionReuseInMis);
    readBool ("indirectOnly", config.indirectOnly);
    readBool ("emissiveGeometry", config.emissiveGeometry);
    readUint ("guidingDebugView", config.guidingDebugView);
    readUint ("treeWeightMode", config.treeWeightMode);
    readFloat("defaultSeconds", config.defaultSeconds);
    if (doc.HasMember("outputDir") && doc["outputDir"].IsString())
        config.outputDir = doc["outputDir"].GetString();

    // A key this parser does not know is silently ignored, which is how
    // "injectionReuse": true sat in the evaluation config for weeks reading like a
    // record of the conditions a run was measured under while doing nothing at all.
    // A config file IS the record of a measurement's conditions, so a key that does
    // not reach the renderer has to be loud.
    static constexpr const char* kKnownKeys[] = {
        "width", "height", "spp", "bounces", "postProcessEnabled", "exposure", "contrast",
        "saturation", "lift", "indirectSkyClamp", "skyLighting", "injectionReuseInMis",
        "indirectOnly", "emissiveGeometry", "guidingDebugView", "treeWeightMode", "defaultSeconds",
        "outputDir",
        "lights",
    };
    for (auto member = doc.MemberBegin(); member != doc.MemberEnd(); ++member)
    {
        const std::string name = member->name.GetString();
        bool known = false;
        for (const char* key : kKnownKeys)
            known = known || name == key;
        if (!known)
            spdlog::warn("Headless config {}: key '{}' is not recognised and does nothing", path, name);
    }

    if (doc.HasMember("lights") && doc["lights"].IsArray())
    {
        config.lightsSpecified = true;

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
