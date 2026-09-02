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

    // Supplemental Sec. 2's biased shortcut (vxpg.injection.reuseInMis). Default
    // false = the integrator traces its own BSDF sample and stays unbiased.
    bool injectionReuseInMis = false;

    // Indirect illumination only (pathtracing.indirectOnly). It lives here rather
    // than only on the command line because it changes WHICH INTEGRAL is rendered:
    // a reference image and the arms scored against it must share it, and a
    // mismatch shows up as both arms being equally wrong with nothing in the table
    // to reveal it. Making it a property of the config file makes it a property of
    // the (scene, light rig) pair, which is exactly the unit a reference belongs to.
    bool indirectOnly = false;

    // Emissive triangles light the scene (pathtracing.emissiveGeometry). Here for the
    // same reason as indirectOnly: switching the scene's own emitters off changes
    // which integral is rendered, so the reference image and every arm scored against
    // it have to agree on it, and it belongs to the (scene, light rig) pair.
    bool emissiveGeometry = true;

    float       defaultSeconds = 5.0f;
    std::string outputDir      = "SavedUserData/Screenshots";

    // When the config names a "lights" array, it replaces the scene's glTF/default
    // lights for the run — an EMPTY array included, which is how a measurement asks
    // for a scene lit by its emissive geometry alone. Key absent => keep whatever
    // lights the loaded scene provides.
    std::vector<HeadlessLight> lights;
    bool                       lightsSpecified = false;
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
    // give. They really are independent now that nothing is carried between frames:
    // the guide is rebuilt from scratch every frame, so two images differ only by
    // their RNG stream.
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

    // --settle N: frames pumped BETWEEN images, outside every measured window. The capture
    // of one image leaves work behind — the readback flushes the queue and the encode is
    // handed to writer threads — and part of that lands in the next frame's clock delta.
    // The engine already subtracts what it can measure of it, and at a 30 s budget the
    // remainder is invisible; at a budget of one display frame it is not. Measured on
    // zero-day at 24 ms: an image that should hold two frames of 12.1 ms held one frame of
    // 36.3 ms. Zero by default, because a settle frame is wrong for a metric defined on
    // CONSECUTIVE frames (temporal error), and right for INDEPENDENT images.
    uint32_t settleFrames = 0;

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

    // --levers a,b (or "none"): the complete set of vendor levers to enable
    // (ADR 0020). Stating the whole set rather than a delta is what keeps one row
    // of a lever matrix from inheriting the previous row's state.
    std::vector<std::string> levers;
    bool                     leversSpecified = false;

    // --config <path>: which headless config to read. It carries the render settings
    // AND the lights, so a scene needs the one lit for it — the default file lights a
    // small interior and leaves Sponza black.
    std::string configPath;

    // --cvar-matrix "renderer.numBounces=1,2,4;vxpg.treeWeightMode=0,1": a sweep over
    // the CROSS PRODUCT of these, measured inside ONE process. These are runtime
    // CVars — no shader recompile, no pipeline rebuild — so a settings point costs a
    // re-arm rather than a process launch, and at short budgets the launch was 95% of
    // the wall clock. Each capture records the point it belongs to in its sidecar.
    std::string cvarMatrix;
};

// One "renderer.numBounces=2" assignment, applied to whichever CVar type exists.
// Shared by --cvar and the matrix sweep so both resolve types the same way.
void ApplyCVarAssignment(const std::string& name, const std::string& value);

// Applies --cvar and --levers. Called once at startup and again by a headless run
// after its config file has been read, because the config writes CVars too.
void ApplyCommandLineOverrides(const HeadlessArgs& args);

HeadlessArgs   ParseHeadlessArgs(int argc, wchar_t** argv);
HeadlessConfig LoadHeadlessConfig(const std::string& path);
