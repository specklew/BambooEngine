#include "pch.h"
#include "VendorLevers.h"

#include "Utils/CVars.h"

#include <algorithm>
#include <sstream>

VendorLevers& VendorLevers::Get()
{
    static VendorLevers instance;
    return instance;
}

VendorLevers::VendorLevers()
{
    // Both of these existed as hand-rolled sidecar variants before the registry did
    // (guidedPathTracing.rg.clean.shader, .rg.onesample.shader). They are the same
    // mechanism, so they live here now — and they double as the proof that the
    // generic path reproduces what the measured one did.
    m_levers = {
        { "noviews", "renderer.raygenCleanVariant",
          "GUIDING_DEBUG_VIEWS=0 RT_DEBUG_VIEWS=0", LeverScope::ShaderVariant, true, false, nullptr, nullptr,
          "Compile the raygen without debug-view code. Re-measured 2026-08-23 on this RDNA driver: a wash "
          "on PT and ~2 % FASTER on VXPG, reversing the 2026-08 reading that made it opt-in (ADR 0014)." },
        { "swizzle", "renderer.raygenSwizzle",
          "RAYGEN_SWIZZLE=1", LeverScope::ShaderVariant, false, false, nullptr, nullptr,
          "Morton launch-to-pixel swizzle inside a 32x32 tile (ADR 0020 R2). Pads the dispatch; bit-exact." },
        // Wave size reaches the inline-RayQuery integrator only: [WaveSize] is a
        // compute/node attribute in HLSL, so the DXR pipeline raygen cannot take it
        // (ADR 0020 R10). Inert unless vxpg.integrator.inlineRq selects that backend.
        { "wave32", "renderer.forceWave32",
          "FORCED_WAVE_SIZE=32", LeverScope::ShaderVariant, false, false, nullptr, "wavesize",
          "Force wave32 on the inline-RayQuery integrator (ADR 0020 R10)." },
        { "wave64", "renderer.forceWave64",
          "FORCED_WAVE_SIZE=64", LeverScope::ShaderVariant, false, false, nullptr, "wavesize",
          "Force wave64 on the inline-RayQuery integrator (ADR 0020 R10)." },
        // The "dxrprofile" group: each of these compiles the DXR libraries at a
        // different shader profile, and a state object has exactly one. allShaders
        // because the payload declaration they change is shared (ADR 0020 R7).
        //
        // `payloadqual` (lib_6_7 + qualifiers) and its `lib66` control were measured on
        // 2026-08-21 and removed: −2.9 % frames on PT and −16.6 % on VXPG, with the
        // control showing most of that was the profile rather than the qualifiers. The
        // PAQ macros and the payload annotations stay — lib_6_9 makes them mandatory,
        // so `ser` needs them.
        //
        // SER (ADR 0020 R1). lib_6_9 makes the payload annotation mandatory, so this
        // necessarily carries the qualifier define too — which is why the lever below
        // exists as its control: lib_6_9 WITHOUT the reorder call.
        { "ser", "renderer.shaderExecutionReordering",
          "PAYLOAD_QUALIFIERS=1 RAYGEN_SER=1", LeverScope::ShaderVariant, false, true, "lib_6_9", "dxrprofile",
          "dx::MaybeReorderThread between traversal and shading (ADR 0020 R1). Needs SM 6.9." },
        { "legacysolidangle", "renderer.legacySolidAngle",
          "GUIDE_LEGACY_SOLID_ANGLE=1", LeverScope::ShaderVariant, false, false, nullptr, nullptr,
          "Restore the spherical-excess solid angle: three full SphericalQuadInit per guided sample "
          "instead of one triple product each. The A/B control for that change, not an optimization." },
        { "lib69", "renderer.forceLib69",
          "PAYLOAD_QUALIFIERS=1", LeverScope::ShaderVariant, false, true, "lib_6_9", "dxrprofile",
          "Compile the DXR libraries as lib_6_9 without the reorder call (ADR 0020 R1 control)." },
    };
}

const VendorLever* VendorLevers::Find(const std::string& name) const
{
    const auto match = std::find_if(m_levers.begin(), m_levers.end(),
        [&](const VendorLever& lever) { return name == lever.name; });
    return match == m_levers.end() ? nullptr : &*match;
}

bool VendorLevers::IsEnabled(const VendorLever& lever) const
{
    const int32_t* value = CVarSystem::Get()->GetIntCVar(StringId(lever.cvarName));
    return value != nullptr && *value != 0;
}

void VendorLevers::SetEnabled(const std::string& name, bool enabled) const
{
    const VendorLever* lever = Find(name);
    if (lever == nullptr)
    {
        spdlog::error("Unknown vendor lever '{}'", name);
        return;
    }
    CVarSystem::Get()->SetCVarInt(StringId(lever->cvarName), enabled ? 1 : 0);
}

std::string VendorLevers::VariantKey(bool debugViewActive) const
{
    std::vector<std::string> names;
    for (const VendorLever& lever : m_levers)
    {
        if (lever.scope != LeverScope::ShaderVariant || !IsEnabled(lever))
            continue;
        if (debugViewActive && lever.suppressedByDebugView)
            continue;
        if (lever.exclusiveGroup != nullptr)
        {
            const auto sameGroupAndOn = [&](const VendorLever& other) {
                return &other != &lever && other.exclusiveGroup != nullptr &&
                       std::string(other.exclusiveGroup) == lever.exclusiveGroup && IsEnabled(other);
            };
            const auto clash = std::find_if(m_levers.begin(), m_levers.end(), sameGroupAndOn);
            if (clash != m_levers.end())
            {
                spdlog::error("Levers '{}' and '{}' are both '{}'; both dropped", lever.name, clash->name, lever.exclusiveGroup);
                continue;
            }
        }
        names.emplace_back(lever.name);
    }

    std::sort(names.begin(), names.end()); // the key is an identity, not an order of events
    std::string key;
    for (const std::string& name : names)
    {
        if (!key.empty())
            key += '+';
        key += name;
    }
    return key;
}

std::string VendorLevers::AllShaderSubsetKey(const std::string& key)
{
    if (key.empty())
        return {};

    const VendorLevers& levers = Get();
    std::string subset;
    std::stringstream stream(key);
    std::string name;
    while (std::getline(stream, name, '+'))
    {
        const VendorLever* lever = levers.Find(name);
        if (lever == nullptr || !lever->allShaders)
            continue;
        if (!subset.empty())
            subset += '+';
        subset += name;
    }
    return subset;
}

std::string VendorLevers::TargetForKey(const std::string& key)
{
    if (key.empty())
        return {};

    const VendorLevers& levers = Get();
    std::string target;
    std::stringstream stream(key);
    std::string name;
    while (std::getline(stream, name, '+'))
    {
        const VendorLever* lever = levers.Find(name);
        if (lever == nullptr || lever->targetOverride == nullptr)
            continue;
        if (target.empty())
            target = lever->targetOverride;
        else if (target != lever->targetOverride)
            spdlog::error("Levers in key '{}' demand different targets ('{}' kept)", key, target);
    }
    return target;
}

std::string VendorLevers::DefinesForKey(const std::string& key)
{
    if (key.empty())
        return {};

    const VendorLevers& levers = Get();
    std::string defines;
    std::stringstream stream(key);
    std::string name;
    while (std::getline(stream, name, '+'))
    {
        const VendorLever* lever = levers.Find(name);
        if (lever == nullptr)
        {
            spdlog::error("Shader variant key names unknown lever '{}'", name);
            continue;
        }
        std::stringstream tokens(lever->defines);
        std::string token;
        while (tokens >> token)
        {
            if (!defines.empty())
                defines += ' ';
            defines += "-D ";
            defines += token;
        }
    }
    return defines;
}

std::string VendorLevers::VariantAsset(const std::string& basePath, const std::string& key)
{
    if (key.empty())
        return basePath;

    // Before the extension, not after it. A ResourceId is the asset path with its
    // extension stripped, so a suffix past the dot strips away with it and every
    // variant collapses onto the base shader's id — which is exactly what happened:
    // the pipeline reported it was built with a variant while running base DXIL.
    const size_t dot = basePath.find_last_of('.');
    if (dot == std::string::npos)
        return basePath + "|" + key;
    return basePath.substr(0, dot) + "|" + key + basePath.substr(dot);
}

std::string VendorLevers::ActiveNames() const
{
    std::string names;
    for (const VendorLever& lever : m_levers)
    {
        if (!IsEnabled(lever))
            continue;
        if (!names.empty())
            names += ',';
        names += lever.name;
    }
    return names;
}
