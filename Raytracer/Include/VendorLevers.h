#pragma once

#include <string>
#include <vector>

// One switchable optimization, and the one place that says what switching it costs
// (ADR 0020). The study this exists for is a grid of combinations, so every lever
// has to be flippable on its own, from ImGui and from a command line, and every
// measurement has to record which ones were on.
enum class LeverScope
{
    Runtime,       // a bind or a dispatch shape: flip it and the next frame differs
    ShaderVariant, // changes hot-kernel code: needs a recompile + pipeline rebuild
    Resource,      // needs GPU resources rebuilt (BLAS flags, for instance)
};

struct VendorLever
{
    // Lowercase, and it ends up inside a shader asset id — see VariantKey.
    const char* name;
    // The CVar that actually holds the state. Levers point at existing CVars where
    // one already exists, so there is never a second source of truth for a switch.
    const char* cvarName;
    // ShaderVariant only: the defines this lever adds, without "-D". A lever may
    // name defines that a given shader does not use (the debug-view flags differ
    // between the path tracer and the guided integrator); an unused define is free.
    const char* defines;
    LeverScope  scope;
    // A lever whose stripped code an active debug view needs. Forced off while one
    // is selected, because the alternative is a view that renders nothing.
    bool        suppressedByDebugView;
    // false = the raygen carries this lever alone (the default: that is the kernel
    // whose codegen shape moves, and recompiling the rest buys nothing). true = every
    // shader in the state object gets it, for levers that change a SHARED declaration
    // — payload qualifiers must match across raygen, closest hit and miss or the
    // libraries disagree about the payload they pass each other.
    bool        allShaders;
    // Shader target this lever requires, or nullptr to keep each shader's own. A
    // lever needs this when the feature it turns on is gated by the profile rather
    // than by a define — payload qualifiers are mandatory at lib_6_7, so the base
    // shaders stay at lib_6_5 and the variant raises the target with the annotations.
    const char* targetOverride;
    // Levers sharing a group name are mutually exclusive: they set the same define
    // or the same shader profile, so two of them at once is not a combination, it is
    // a contradiction. All of them are dropped and it is logged — an impossible
    // request must not quietly measure as if it were one of its halves.
    const char* exclusiveGroup;
    const char* description;
};

class VendorLevers
{
public:
    static VendorLevers& Get();

    const std::vector<VendorLever>& All() const { return m_levers; }
    const VendorLever*              Find(const std::string& name) const;

    bool IsEnabled(const VendorLever& lever) const;
    void SetEnabled(const std::string& name, bool enabled) const;

    // "noviews+onesample" — the enabled ShaderVariant levers, sorted, '+'-joined.
    // Empty when none are on, which is what keeps the default build addressing the
    // plain, unsuffixed shader asset.
    std::string VariantKey(bool debugViewActive) const;

    // The part of a key that every shader in the state object must be built with.
    // Empty for the common case, which leaves the non-raygen shaders on their plain
    // asset ids and their existing blobs.
    static std::string AllShaderSubsetKey(const std::string& key);

    // Target a key's levers demand ("lib_6_7"), or empty to keep the shader's own.
    // Two levers in one key demanding different targets is a configuration error and
    // is logged; the first one wins so the build stays deterministic.
    static std::string TargetForKey(const std::string& key);

    // "-D GUIDING_DEBUG_VIEWS=0" for a key produced above.
    // Static because the shader loader resolves a key long after the frame that
    // built it, and must not depend on the levers' state having stayed put.
    static std::string DefinesForKey(const std::string& key);

    // "resources/shaders/x.rg.shader" + "noviews" -> "resources/shaders/x.rg|noviews.shader".
    // The key goes BEFORE the extension; see the definition for why. An empty key
    // returns the base path unchanged.
    static std::string VariantAsset(const std::string& basePath, const std::string& key);

    // Comma-separated names, for a capture sidecar. Includes Runtime and Resource
    // levers: a measurement has to record every switch, not just the compiled ones.
    std::string ActiveNames() const;

private:
    VendorLevers();
    std::vector<VendorLever> m_levers;
};
