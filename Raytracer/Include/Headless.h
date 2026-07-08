#pragma once

#include <cstdint>
#include <string>
#include <vector>

// One light from the headless config. Plain primitives so this header stays free of
// engine/DirectX types; HeadlessRunner converts these into the engine's LightData.
struct HeadlessLight
{
    std::string type = "directional";   // directional | point | spot
    float position[3]  = { 0.0f, 0.0f, 0.0f };
    float direction[3] = { 0.0f, -1.0f, 0.0f };
    float color[3]     = { 1.0f, 1.0f, 1.0f };
    float intensity    = 3.0f;
    float range        = 0.0f;
};

// Stable render defaults for headless mode, loaded from SavedUserData/headless.json.
// Flags override these; these override the engine's built-in CVar defaults.
struct HeadlessConfig
{
    uint32_t width  = 1280;
    uint32_t height = 720;

    uint32_t spp     = 1;
    uint32_t bounces = 1;

    bool  postProcessEnabled = true;
    float exposure   = 1.0f;
    float contrast   = 1.0f;
    float saturation = 1.0f;
    float lift       = 0.0f;

    // Indirect skybox firefly clamp (pathtracing.indirectSkyClamp). 0 = disabled.
    // Set >0 for benchmark convergence; applied identically to PT and VXPG.
    float indirectSkyClamp = 0.0f;

    float       defaultSeconds = 5.0f;
    std::string outputDir      = "SavedUserData/Screenshots";

    // When non-empty, these replace the scene's glTF/default lights for the run.
    // Empty => keep whatever lights the loaded scene provides.
    std::vector<HeadlessLight> lights;
};

// Per-run intent parsed from the command line.
struct HeadlessArgs
{
    bool headless = false;

    std::string              scene;       // glTF path, or a bare model name resolved under resources/models/
    std::vector<std::string> places;      // saved place names within the scene
    std::vector<std::string> techniques;  // raytracing technique registry names

    float       seconds = -1.0f;          // < 0 => use config default
    std::string outDir;                   // empty => use config output dir
};

HeadlessArgs   ParseHeadlessArgs(int argc, wchar_t** argv);
HeadlessConfig LoadHeadlessConfig(const std::string& path);
