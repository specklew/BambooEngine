#include "pch.h"
#include "HeadlessRunner.h"

#include "Renderer.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <unordered_map>

namespace
{
    std::string RunFolderTimestamp()
    {
        std::time_t t = std::time(nullptr);
        struct tm tm_info = {};
        localtime_s(&tm_info, &t);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm_info);
        return buf;
    }

    std::wstring ResolveScenePath(const std::string& scene)
    {
        std::string path = scene;
        const bool looksLikePath = scene.find('/') != std::string::npos ||
                                   scene.find('\\') != std::string::npos ||
                                   scene.find(".gl") != std::string::npos;
        if (!looksLikePath)
            path = "resources/models/" + scene + ".glb";
        return std::wstring(path.begin(), path.end());
    }

    bool Contains(const std::vector<std::string>& haystack, const std::string& needle)
    {
        return std::find(haystack.begin(), haystack.end(), needle) != haystack.end();
    }

    void LogAvailable(const char* label, const std::vector<std::string>& names)
    {
        std::string joined;
        for (const std::string& n : names) { if (!joined.empty()) joined += ", "; joined += n; }
        spdlog::error("Available {}: {}", label, joined.empty() ? "<none>" : joined);
    }
}

HeadlessRunner::HeadlessRunner(Renderer& renderer, HeadlessArgs args, HeadlessConfig config)
    : m_renderer(renderer), m_args(std::move(args)), m_config(std::move(config))
{
}

void HeadlessRunner::PumpFrame()
{
    MSG msg = {};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    m_clock.Tick();
    m_renderer.Update(m_clock.GetDeltaSeconds(), m_clock.GetTotalSeconds());
    m_renderer.Render(m_clock.GetDeltaSeconds(), m_clock.GetTotalSeconds());
}

void HeadlessRunner::ApplyConfiguredLights()
{
    if (m_config.lights.empty())
        return; // keep the scene's own glTF/default lights

    const std::unordered_map<std::string, LightType> typeByName = {
        { "directional", LightType::Directional },
        { "point",       LightType::Point },
        { "spot",        LightType::Spot },
    };

    std::vector<LightData> lights;
    lights.reserve(m_config.lights.size());
    for (const HeadlessLight& source : m_config.lights)
    {
        const auto it = typeByName.find(source.type);
        if (it == typeByName.end())
        {
            spdlog::warn("Headless light has unknown type '{}', skipping", source.type);
            continue;
        }

        LightData light{};
        light.type      = it->second;
        light.position  = { source.position[0],  source.position[1],  source.position[2] };
        light.direction = { source.direction[0], source.direction[1], source.direction[2] };
        light.color     = { source.color[0],     source.color[1],     source.color[2] };
        light.intensity = source.intensity;
        light.range     = source.range;
        lights.push_back(light);
    }

    spdlog::info("Headless config supplies {} light(s), overriding scene lights", lights.size());
    m_renderer.SetLights(lights);
}

bool HeadlessRunner::Validate() const
{
    if (m_args.places.empty() || m_args.techniques.empty())
    {
        spdlog::error("Headless run needs at least one --places and one --techniques entry");
        return false;
    }

    const std::vector<std::string> validTechniques = m_renderer.GetTechniqueNames();
    const std::vector<std::string> validPlaces     = m_renderer.GetPlaceNames();

    bool ok = true;
    for (const std::string& technique : m_args.techniques)
        if (!Contains(validTechniques, technique))
        {
            spdlog::error("Unknown technique '{}'", technique);
            ok = false;
        }
    for (const std::string& place : m_args.places)
        if (!Contains(validPlaces, place))
        {
            spdlog::error("Unknown place '{}' in this scene", place);
            ok = false;
        }

    if (!ok)
    {
        LogAvailable("techniques", validTechniques);
        LogAvailable("places", validPlaces);
    }
    return ok;
}

int HeadlessRunner::Run()
{
    m_renderer.SetHeadless(true);
    m_renderer.ApplyRenderConfig(m_config);
    m_renderer.SetRaytracing(true);

    m_renderer.LoadScene(ResolveScenePath(m_args.scene));
    ApplyConfiguredLights();

    if (!Validate())
        return 2;

    const float seconds = m_args.seconds >= 0.0f ? m_args.seconds : m_config.defaultSeconds;
    const std::string baseDir = m_args.outDir.empty() ? m_config.outputDir : m_args.outDir;
    const std::string runDir  = baseDir + "/run-" + RunFolderTimestamp();
    const std::string model   = std::filesystem::path(m_args.scene).stem().string();

    spdlog::info("Headless run: {} captures into {} ({:.1f}s each)",
                 m_args.places.size() * m_args.techniques.size(), runDir, seconds);

    for (const std::string& technique : m_args.techniques)
    {
        m_renderer.SetTechnique(technique);
        for (const std::string& place : m_args.places)
        {
            m_renderer.GoToPlace(place);
            PumpFrame(); // absorb setup/technique-init time so the first armed frame has a normal delta

            m_renderer.ArmScreenshot(seconds, model, place, runDir, place + "-" + technique);
            while (!m_renderer.ScreenshotIdle())
                PumpFrame();

            spdlog::info("Captured {}-{}", place, technique);
        }
    }

    spdlog::info("Headless run complete");
    return 0;
}
