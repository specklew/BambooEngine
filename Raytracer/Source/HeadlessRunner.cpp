#include "pch.h"
#include "HeadlessRunner.h"

#include "Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <unordered_map>
#include <vector>

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

    bool LooksLikePath(const std::string& scene)
    {
        return scene.find('/')  != std::string::npos ||
               scene.find('\\') != std::string::npos ||
               scene.find(".gl") != std::string::npos ||
               scene.find(".json") != std::string::npos; // Tungsten scene
    }

    std::wstring ResolveScenePath(const std::string& scene)
    {
        std::string path = scene;
        if (!LooksLikePath(scene))
        {
            path = "resources/models/" + scene + ".glb";
            if (!std::filesystem::exists(path))
            {
                // Research scenes sit one folder down and rewrite dashes: veach-ajar -> veach-ajar/veach_ajar_core.glb.
                std::string fileStem = scene;
                std::replace(fileStem.begin(), fileStem.end(), '-', '_');
                const std::string research = "resources/models/gltf_research_scenes/" + scene + "/" + fileStem + "_core.glb";
                if (std::filesystem::exists(research))
                {
                    spdlog::info("Scene '{}' resolved to research scene {}", scene, research);
                    path = research;
                }
            }
        }
        return std::wstring(path.begin(), path.end());
    }

    // A bare --scene name IS the scene's identity, so it keys states.json directly.
    // That keeps research scenes (whose file is veach_ajar_core.glb) on the camera
    // set saved as "veach-ajar", and is a no-op for flat models where the two agree.
    std::string ResolveStatesKey(const HeadlessArgs& args)
    {
        if (!args.statesKey.empty())
            return args.statesKey;
        return LooksLikePath(args.scene) ? std::string{} : args.scene;
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

// Discards frames until the frame time settles, so a measurement reports the
// technique rather than the GPU's clock ramp. Two things are being waited on at
// once: boost clocks settling under sustained load, and the one-time costs (PSO
// creation, BVH build, first dispatch of every node). The guide itself needs no
// warm-up — it is rebuilt from scratch every frame and carries nothing forward.
float HeadlessRunner::WarmUp()
{
    // Always spend the one-time costs first, and spend them OUTSIDE the timed part
    // below. A compile-time vendor lever swaps the raygen variant, which routes
    // through the full OnShaderReload (DXC + state object + SBT) and can cost
    // seconds; counted as elapsed warm-up it blows straight through the cap, the
    // loop exits without ever settling, and the reload's cost then lands in the
    // first armed frame — which was measured as a one-frame capture on every
    // lever combination.
    for (int i = 0; i < 16; ++i)
        PumpFrame();

    if (m_args.warmupSeconds <= 0.0f)
        return 0.0f;

    constexpr size_t kWindow      = 30;   // frames the stability test looks at
    constexpr double kMaxVariation = 0.02; // coefficient of variation to call it settled
    constexpr double kMinSeconds  = 1.0;  // never declare stability off a handful of frames

    std::vector<double> window;
    window.reserve(kWindow);
    double elapsed = 0.0;
    size_t next    = 0;

    while (elapsed < m_args.warmupSeconds)
    {
        PumpFrame();
        const double delta = m_clock.GetDeltaSeconds();
        elapsed += delta;

        if (window.size() < kWindow) window.push_back(delta);
        else { window[next] = delta; next = (next + 1) % kWindow; }

        if (window.size() < kWindow || elapsed < kMinSeconds)
            continue;

        double sum = 0.0;
        for (double d : window) sum += d;
        const double mean = sum / window.size();
        double variance = 0.0;
        for (double d : window) variance += (d - mean) * (d - mean);
        const double coefficientOfVariation = mean > 0.0 ? std::sqrt(variance / window.size()) / mean : 0.0;

        if (coefficientOfVariation < kMaxVariation)
        {
            spdlog::info("Warm-up settled after {:.2f}s at {:.3f} ms/frame (CV {:.3f})",
                         elapsed, mean * 1000.0, coefficientOfVariation);
            return static_cast<float>(elapsed);
        }
    }

    spdlog::warn("Warm-up hit its {:.2f}s cap without settling — frame time is still moving",
                 m_args.warmupSeconds);
    return static_cast<float>(elapsed);
}

// --checkpoints log:K | every:N | list:a,b,c, in the budget's own unit. Always
// ends at the budget, so the last image of a schedule is the one an unchecked run
// would have produced — a curve run and a point run stay comparable.
CaptureSchedule HeadlessRunner::BuildSchedule() const
{
    const CaptureBudget budget = m_args.budgetFrames > 0
        ? CaptureBudget::Frames(m_args.budgetFrames)
        : CaptureBudget::Seconds(m_args.seconds >= 0.0f ? m_args.seconds : m_config.defaultSeconds);

    if (m_args.checkpoints.empty())
        return CaptureSchedule::AtEnd(budget);

    const size_t      colon = m_args.checkpoints.find(':');
    const std::string kind  = m_args.checkpoints.substr(0, colon);
    const std::string value = colon == std::string::npos ? std::string{} : m_args.checkpoints.substr(colon + 1);

    std::vector<double> points;
    if (kind == "log")
    {
        // Log spacing, because convergence goes as 1/sqrt(N): equal steps on a log
        // axis are equal steps of visible improvement, where linear ones crowd
        // every interesting point into the first tenth of the run.
        const int count = std::max(2, std::atoi(value.c_str()));
        const double first = budget.kind == CaptureBudget::Kind::Frames ? 1.0 : budget.value / 100.0;
        for (int i = 0; i < count; ++i)
        {
            const double t = static_cast<double>(i) / (count - 1);
            points.push_back(first * std::pow(budget.value / first, t));
        }
    }
    else if (kind == "every")
    {
        const double step = std::max(1.0, std::atof(value.c_str()));
        for (double p = step; p < budget.value; p += step)
            points.push_back(p);
        points.push_back(budget.value);
    }
    else if (kind == "list")
    {
        std::stringstream ss(value);
        std::string item;
        while (std::getline(ss, item, ','))
            if (!item.empty())
                points.push_back(std::atof(item.c_str()));
    }
    else
    {
        // Falling back to a single capture at the end would quietly answer a different
        // question than the one asked — the same class of silent substitution as a
        // malformed --budget, which used to run on the default budget and exit 0.
        spdlog::error("--checkpoints expects log:K, every:N or list:a,b,c — got '{}'. Refusing to "
                      "fall back to a single end-of-run capture, because that is a different measurement",
                      m_args.checkpoints);
        std::exit(2);
    }

    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.empty() || points.back() < budget.value)
        points.push_back(budget.value);

    return { budget, points };
}

// "renderer.numBounces=1,2,4;vxpg.treeWeightMode=0,1" -> 6 points, first dimension
// varying slowest. Order matters for more than tidiness: the slowest-varying axis is
// the one whose neighbours share a reference image, so a sweep can be scored without
// re-rendering ground truth for every point.
std::vector<HeadlessRunner::SettingsPoint> HeadlessRunner::ExpandSettingsMatrix() const
{
    std::vector<SettingsPoint> points{ SettingsPoint{} };
    if (m_args.cvarMatrix.empty())
        return points;

    std::stringstream dimensions(m_args.cvarMatrix);
    std::string dimension;
    while (std::getline(dimensions, dimension, ';'))
    {
        if (dimension.empty())
            continue;

        const size_t equals = dimension.find('=');
        if (equals == std::string::npos)
        {
            spdlog::error("--cvar-matrix dimension '{}' is not name=v1,v2", dimension);
            continue;
        }

        const std::string name = dimension.substr(0, equals);
        std::vector<std::string> values;
        std::stringstream valueList(dimension.substr(equals + 1));
        std::string value;
        while (std::getline(valueList, value, ','))
            if (!value.empty())
                values.push_back(value);

        if (values.empty())
        {
            spdlog::error("--cvar-matrix dimension '{}' lists no values", name);
            continue;
        }

        std::vector<SettingsPoint> expanded;
        expanded.reserve(points.size() * values.size());
        for (const SettingsPoint& point : points)
            for (const std::string& one : values)
            {
                SettingsPoint next = point;
                next.assignments.emplace_back(name, one);
                next.tag += (next.tag.empty() ? "" : ";") + name + "=" + one;
                expanded.push_back(std::move(next));
            }
        points.swap(expanded);
    }

    return points;
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
    if (!m_config.lightsSpecified)
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

    // No --debug-views means "whatever the config asked for", NOT view 0: hardcoding 0
    // here silently overwrote `guidingDebugView` from the render config, so a run that
    // asked for the symmetric baseline measured the full guided integrator instead.
    if (m_args.debugViews.empty())
        return { {static_cast<int>(m_config.guidingDebugView), "config"} };

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
    // After the config, not before: it writes the same CVars, so a command line
    // that lost to it would be a silent misconfiguration of a measurement.
    ApplyCommandLineOverrides(m_args);

    m_renderer.LoadScene(ResolveScenePath(m_args.scene), ResolveStatesKey(m_args));

    if (!Validate())
        return 2;

    const CaptureSchedule schedule = BuildSchedule();
    const std::vector<SettingsPoint> settingsPoints = ExpandSettingsMatrix();
    const std::string baseDir = m_args.outDir.empty() ? m_config.outputDir : m_args.outDir;
    const std::string runDir  = baseDir + "/run-" + RunFolderTimestamp();
    const std::string model   = std::filesystem::path(m_args.scene).stem().string();

    spdlog::info("Headless run: {} configuration(s) x {} image(s) into {} (budget {} {}, {} checkpoint(s))",
                 m_args.states.size() * m_args.techniques.size() * settingsPoints.size(), m_args.images, runDir,
                 schedule.budget.value,
                 schedule.budget.kind == CaptureBudget::Kind::Frames ? "frames" : "seconds",
                 schedule.checkpoints.size());

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
            // After the state, not before: a state that carries a `lights` array replaces
            // the light set on entry, so config lights applied earlier were silently
            // discarded — and an empty array in the state wiped them without a word.
            ApplyConfiguredLights();

            for (const RenderTechnique::DebugView& view : views)
            {
                m_renderer.SetTechniqueDebugView(view.index);

              size_t pointIndex = 0;
              for (const SettingsPoint& point : settingsPoints)
              {
                // Runtime CVars only, so a settings point costs a re-warm rather than
                // a process launch; anything needing a pipeline rebuild belongs in
                // --levers, which is a separate axis for exactly that reason.
                for (const auto& [name, value] : point.assignments)
                    ApplyCVarAssignment(name, value);
                m_renderer.SetSettingsTag(point.tag);

                // Warm up before arming. One frame is NOT enough: a slow first frame
                // (technique switch rebuilding the VXPG passes, PSO creation, first
                // dispatch) costs seconds, and that cost lands in the NEXT frame's
                // clock delta — so the first armed frame would read a huge delta and
                // trigger an immediate single-frame capture (observed as frameIndex 0
                // for VXPG at --seconds 1). A view switch can also swap the raygen
                // variant, which rebuilds the pipeline — same reason, same warm-up.
                const float warmupSpent = WarmUp();

                // A sweep writes several points into one run folder, so the point's
                // index joins the name. The index alone is not the record of what was
                // measured — the sidecar's settings string is — but two points must
                // not write the same file.
                std::string stem = m_args.debugViews.empty()
                    ? place + "-" + technique
                    : place + "-" + technique + "-" + view.name;
                if (settingsPoints.size() > 1)
                    stem += fmt::format("-c{:03}", pointIndex);

                // Independent images, one process: accumulation resets between them
                // while the frame counter — and so the per-pixel RNG stream — runs on,
                // which is what makes their spread a real error bar.
                //
                // The images are CONSECUTIVE frames: nothing is rendered between them.
                // There used to be a throwaway frame here to absorb the PNG encode,
                // which the clock would otherwise hand to the next frame — but the
                // encode moved to writer threads and ScreenshotManager already times
                // its own readback and hands it to Renderer::Update to subtract
                // (ConsumeLastCaptureCostSeconds). The throwaway had become a
                // leftover that only made every second frame invisible, which is
                // exactly the wrong thing for a metric defined on CONSECUTIVE frames
                // (temporal error / temporal MSE).
                //
                // Capturing EVERY frame is not free, and the subtraction above does
                // not fully hide it: measured on veach-ajar / VXPG, a run whose every
                // frame is captured reports 6.8-10.4 ms against a 4.63 ms warm-up at
                // 1080p, and 1.4-1.8 ms against 0.51 ms at 480x270 — so roughly a
                // fixed ~1 ms of per-capture machinery plus a resolution-dependent
                // readback/encode term. It does not touch the IMAGES (same frame
                // index, same RNG, same accumulation), so a frame-indexed metric is
                // unaffected; but neither meanFrameMs nor accumulatedTime from such a
                // run describes the technique. Take frame cost and equal-time
                // readings from a run that captures rarely.
                for (uint32_t image = 0; image < m_args.images; ++image)
                {
                    const std::string imageStem = m_args.images > 1
                        ? fmt::format("{}-i{:04}", stem, image)
                        : stem;

                    m_renderer.ArmScreenshot(schedule, model, place, runDir, imageStem,
                                             image, m_args.images, warmupSpent);
                    while (!m_renderer.ScreenshotIdle())
                        PumpFrame();
                }

                spdlog::info("Captured {} ({} image(s)){}", stem, m_args.images,
                             point.tag.empty() ? std::string{} : " [" + point.tag + "]");
                ++pointIndex;
              }
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

    // The last images are still in the encoder queue; the run is not complete until
    // they are files.
    m_renderer.WaitForScreenshotWrites();

    spdlog::info("Headless run complete");
    return 0;
}
