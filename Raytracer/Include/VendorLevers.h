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
    // Name of a lever that cannot be on at the same time (wave32 vs wave64 set the
    // same define). Both on drops BOTH and logs — an impossible request must not
    // quietly measure as if it were one of the two.
    const char* conflictsWith;
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

    // "-D GUIDING_DEBUG_VIEWS=0 -D ONE_SAMPLE_MIS=1" for a key produced above.
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
