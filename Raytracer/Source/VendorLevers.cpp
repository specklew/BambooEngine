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
          "GUIDING_DEBUG_VIEWS=0 RT_DEBUG_VIEWS=0", LeverScope::ShaderVariant, true, nullptr,
          "Compile the raygen without debug-view code. Measured SLOWER on this RDNA driver (ADR 0014)." },
        { "onesample", "vxpg.oneSampleMis",
          "ONE_SAMPLE_MIS=1", LeverScope::ShaderVariant, false, nullptr,
          "One-sample MIS at the first vertex (ADR 0015). Default off; two-sample is what benchmarks measure." },
        { "swizzle", "renderer.raygenSwizzle",
          "RAYGEN_SWIZZLE=1", LeverScope::ShaderVariant, false, nullptr,
          "Morton launch-to-pixel swizzle inside a 32x32 tile (ADR 0020 R2). Pads the dispatch; bit-exact." },
        // Wave size reaches the inline-RayQuery integrator only: [WaveSize] is a
        // compute/node attribute in HLSL, so the DXR pipeline raygen cannot take it
        // (ADR 0020 R10). Inert unless vxpg.integrator.inlineRq selects that backend.
        { "wave32", "renderer.forceWave32",
          "FORCED_WAVE_SIZE=32", LeverScope::ShaderVariant, false, "wave64",
          "Force wave32 on the inline-RayQuery integrator (ADR 0020 R10)." },
        { "wave64", "renderer.forceWave64",
          "FORCED_WAVE_SIZE=64", LeverScope::ShaderVariant, false, "wave32",
          "Force wave64 on the inline-RayQuery integrator (ADR 0020 R10)." },
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
        if (lever.conflictsWith != nullptr)
        {
            const VendorLever* other = Find(lever.conflictsWith);
            if (other != nullptr && IsEnabled(*other))
            {
                spdlog::error("Levers '{}' and '{}' conflict; both dropped", lever.name, other->name);
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
