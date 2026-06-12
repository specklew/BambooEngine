#include "pch.h"
#include "Techniques/GuidedPathTracingPass.h"

TechniqueDesc GuidedPathTracingPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
            {"resources/shaders/raytracing.rg.shader",          L"RayGen",     ShaderRole::RayGen},
            {"resources/shaders/raytracing.ms.shader",          L"Miss",       ShaderRole::Miss},
            {"resources/shaders/guidedPathTracing.ch.shader",   L"GuidedHit",  ShaderRole::ClosestHit},
            {"resources/shaders/raytracing.ah.shader",          L"AnyHit",     ShaderRole::AnyHit},
            {"resources/shaders/raytracing.shadowhit.shader",   L"ShadowHit",  ShaderRole::AnyHit},
            {"resources/shaders/raytracing.shadowmiss.shader",  L"ShadowMiss", ShaderRole::Miss},
    };
    desc.hitGroups = {
        {L"GuidedHitGroup", L"GuidedHit", L"AnyHit"},
        {L"ShadowHitGroup", L"",          L"ShadowHit"},
    };
    desc.maxPayloadSize    = 8 * sizeof(float);
    desc.maxAttributeSize  = 2 * sizeof(float);
    // Vanilla chain depth + shadow rays at the terminal vertex; extra headroom
    // because the first vertex spawns two sequential MIS subtrees.
    desc.maxRecursionDepth = 10;
    return desc;
}

REGISTER_RAYTRACE_TECHNIQUE("Guided Path Tracing (VXPG)", GuidedPathTracingPass)
