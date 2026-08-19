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

    // Skybox illumination switch (pathtracing.skyLighting). false = sky stays
    // visible as background but lights nothing — benchmark isolation, since the
    // VXPG guide only targets direct-lit surfaces.
    bool skyLighting = true;

    // Guided PT debug view (guiding.debugView enum index; 0 = normal render).
    // Diagnosis captures: 1 = BSDF MIS part, 2 = guide MIS part, 3 = MIS weight
    // false-color, 4 = guided-sample acceptance.
    uint32_t guidingDebugView = 0;

    // Bottom light-tree branch weighting (vxpg.tree.weightMode). 0 = intensity-
    // only (default); 1 = geometry + avg-minmax distance (the paper's SLC term).
    uint32_t treeWeightMode = 0;

    // Second-bounce guiding (vxpg.secondBounce, SIByL second=true). Guides the
    // 2nd vertex too; a 2-bounce guided estimator. Default off; needs bounces>=2.
    bool secondBounce = false;

    // One-sample MIS (vxpg.oneSampleMis, ADR 0015): trace one stochastically-
    // picked strategy per sample instead of both. Default off (two-sample,
    // SIByL-faithful).
    bool oneSampleMis = false;

    // Adaptive per-tile selection probability for one-sample MIS
    // (vxpg.oneSample.adaptiveQ). Only meaningful with oneSampleMis true.
    bool oneSampleAdaptiveQ = true;

    // Injection reuse from GI BSDF samples (vxpg.injection.reuseGiSamples,
    // ADR 0009). false = dedicated injection pass every frame.
    bool injectionReuse = true;

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
    std::string              statesKey;   // --states-key: states.json key when the file name is not it
    std::vector<std::string> states;      // saved state names within the scene
    std::vector<std::string> techniques;  // raytracing technique registry names

    float       seconds = -1.0f;          // < 0 => use config default
    std::string outDir;                   // empty => use config output dir

    // --budget frames:N | seconds:T. Overrides --seconds; a frame budget is the
    // equal-sample-count axis, a time budget the equal-time one.
    uint32_t budgetFrames = 0;            // 0 => the budget is in seconds

    // --images M: M INDEPENDENT images in one process, accumulation reset between
    // them while the frame counter (and so the RNG stream) runs on. The spread
    // across them is the measurement's error bar, which a single capture cannot
    // give. The guide is deliberately NOT reset: VXPG carries temporal state
    // (ADR 0009 injection reuse, adaptive q, superpixels) and we measure it in
    // steady state, so images share a guide and are not fully independent.
    uint32_t images = 1;

    // --checkpoints log:K | every:N | list:a,b,c. Images written DURING one
    // accumulation, in the budget's unit — a convergence curve from a single run,
    // orthogonal to --images. Empty => one image at the end of the budget.
    std::string checkpoints;

    // --warmup T: render and discard the real workload for up to T seconds before
    // the first capture, stopping early once the frame time stops moving. Covers
    // the GPU clock ramp, PSO/BVH one-time costs, and the guide reaching steady
    // state. 0 => the legacy fixed 16-frame pump.
    float warmupSeconds = 0.0f;

    // --debug-views: capture each listed view of the active technique instead of
    // the plain render. Indices belong to that technique's own enumeration ("all"
    // takes every view it declares). Empty => one normal capture per technique.
    // This is what makes a debug view checkable without a human at the window.
    std::vector<std::string> debugViews;

    // Headless normally runs without the debug layer, because its uneven
    // per-submit validation cost makes frame counts meaningless. --debug-layer
    // turns it back on for a correctness run; --rdg-dump asks for the
    // graph/barrier/root-signature dump, fired once per technique after its
    // capture. --rdg-timings adds per-node GPU timestamps and makes that dump
    // carry the node cost table; it costs frames, so it is never a timing run.
    // All three exist so a verification run needs no source edit.
    bool debugLayer = false;
    bool rdgDump    = false;
    bool rdgTimings = false;

    // --cvar name=value, repeatable. Applied after Initialize so CVar defaults
    // cannot undo them. Exists so an A/B over an engine switch is one command
    // line rather than a source edit and a rebuild — which matters because only
    // a same-session alternating A/B survives the GPU's thermal drift.
    std::vector<std::string> cvarAssignments;
};

HeadlessArgs   ParseHeadlessArgs(int argc, wchar_t** argv);
HeadlessConfig LoadHeadlessConfig(const std::string& path);
