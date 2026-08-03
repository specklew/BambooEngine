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
                                   scene.find(".gl") != std::string::npos ||
                                   scene.find(".json") != std::string::npos; // Tungsten scene
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

// Resolves --debug-views against whatever the ACTIVE technique declares, so it
// runs after SetTechnique: a view index means different things to the raster and
// raytracing enumerations. Empty return = the run should abort.
std::vector<RenderTechnique::DebugView> HeadlessRunner::ResolveDebugViews() const
{
    const std::vector<RenderTechnique::DebugView> available = m_renderer.GetTechniqueDebugViews();

    if (m_args.debugViews.empty())
        return { {0, "None"} };

    if (m_args.debugViews.size() == 1 && m_args.debugViews.front() == "all")
        return available;

    std::vector<RenderTechnique::DebugView> selected;
    for (const std::string& requested : m_args.debugViews)
    {
        const auto match = std::find_if(available.begin(), available.end(),
            [&](const RenderTechnique::DebugView& view)
            {
                return view.name == requested || std::to_string(view.index) == requested;
            });
        if (match == available.end())
        {
            spdlog::error("Unknown debug view '{}' for the active technique", requested);
            std::string joined;
            for (const RenderTechnique::DebugView& view : available)
            {
                if (!joined.empty()) joined += ", ";
                joined += fmt::format("{}({})", view.name, view.index);
            }
            spdlog::error("Available debug views: {}", joined);
            return {};
        }
        selected.push_back(*match);
    }
    return selected;
}

bool HeadlessRunner::Validate() const
{
    if (m_args.states.empty() || m_args.techniques.empty())
    {
        spdlog::error("Headless run needs at least one --states and one --techniques entry");
        return false;
    }

    const std::vector<std::string> validTechniques = m_renderer.GetTechniqueNames();
    const std::vector<std::string> validStates     = m_renderer.GetStateNames();

    bool ok = true;
    for (const std::string& technique : m_args.techniques)
        if (!Contains(validTechniques, technique))
        {
            spdlog::error("Unknown technique '{}'", technique);
            ok = false;
        }
    for (const std::string& state : m_args.states)
        if (!Contains(validStates, state))
        {
            spdlog::error("Unknown state '{}' in this scene", state);
            ok = false;
        }

    if (!ok)
    {
        LogAvailable("techniques", validTechniques);
        LogAvailable("states", validStates);
    }
    return ok;
}

int HeadlessRunner::Run()
{
    m_renderer.SetHeadless(true);
    m_renderer.ApplyRenderConfig(m_config);

    m_renderer.LoadScene(ResolveScenePath(m_args.scene));
    ApplyConfiguredLights();

    if (!Validate())
        return 2;

    const float seconds = m_args.seconds >= 0.0f ? m_args.seconds : m_config.defaultSeconds;
    const std::string baseDir = m_args.outDir.empty() ? m_config.outputDir : m_args.outDir;
    const std::string runDir  = baseDir + "/run-" + RunFolderTimestamp();
    const std::string model   = std::filesystem::path(m_args.scene).stem().string();

    spdlog::info("Headless run: {} captures into {} ({:.1f}s each)",
                 m_args.states.size() * m_args.techniques.size(), runDir, seconds);

    for (const std::string& technique : m_args.techniques)
    {
        m_renderer.SetTechnique(technique);
        m_renderer.ResetGraphTimingHistory();

        // One entry per capture. Without --debug-views that is the plain render;
        // with it, each named view of this technique becomes its own capture.
        const std::vector<RenderTechnique::DebugView> views = ResolveDebugViews();
        if (views.empty())
            return 2;

        for (const std::string& place : m_args.states)
        {
            m_renderer.GoToState(place);

            for (const RenderTechnique::DebugView& view : views)
            {
                m_renderer.SetTechniqueDebugView(view.index);

                // Warm up several frames before arming. One frame is NOT enough: a
                // slow first frame (technique switch rebuilding the VXPG passes, PSO
                // creation, first dispatch) costs seconds, and that cost lands in the
                // NEXT frame's clock delta — so the first armed frame would read a
                // huge delta and trigger an immediate single-frame capture (observed
                // as frameIndex 0 for VXPG at --seconds 1). Pumping until frames are
                // cheap keeps the one-time stall out of the timed capture window.
                // A view switch can also swap the raygen variant, which rebuilds the
                // pipeline — same reason, same warmup.
                for (int warmup = 0; warmup < 16; ++warmup)
                    PumpFrame();

                const std::string stem = m_args.debugViews.empty()
                    ? place + "-" + technique
                    : place + "-" + technique + "-" + view.name;

                m_renderer.ArmScreenshot(seconds, model, place, runDir, stem);
                while (!m_renderer.ScreenshotIdle())
                    PumpFrame();

                spdlog::info("Captured {}", stem);
            }
        }

        // One dump per technique, after its captures: the graph is warm, every
        // root signature it needs has been built, and the node cost table covers
        // the whole capture window rather than a single cold frame.
        if (m_args.rdgDump)
        {
            spdlog::info("[RDG] dump for technique '{}'", technique);
            CVarSystem::Get()->SetCVarInt(StringId("rdg.dump"), 1);
            PumpFrame();
        }
    }

    spdlog::info("Headless run complete");
    return 0;
}
