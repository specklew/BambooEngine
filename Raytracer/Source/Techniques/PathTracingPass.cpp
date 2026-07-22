#include "pch.h"
#include "Techniques/PathTracingPass.h"

TechniqueDesc PathTracingPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
            {m_compileDebugViews ? "resources/shaders/raytracing.rg.shader"
                                 : "resources/shaders/raytracing.rg.clean.shader",
                                                                L"RayGen",     ShaderRole::RayGen},
            {"resources/shaders/raytracing.ms.shader",          L"Miss",       ShaderRole::Miss},
            {"resources/shaders/raytracing.ch.shader",          L"Hit",        ShaderRole::ClosestHit},
            {"resources/shaders/raytracing.ah.shader",          L"AnyHit",     ShaderRole::AnyHit},
            {"resources/shaders/raytracing.shadowhit.shader",   L"ShadowHit",  ShaderRole::AnyHit},
            {"resources/shaders/raytracing.shadowmiss.shader",  L"ShadowMiss", ShaderRole::Miss},
    };
    desc.hitGroups = {
        {L"PrimaryHitGroup", L"Hit", L"AnyHit"},
        {L"ShadowHitGroup",  L"",    L"ShadowHit"},
    };
    desc.maxPayloadSize    = 5 * sizeof(uint32_t); // PtPayload: 3x uint + float2 (ADR 0007)
    desc.maxAttributeSize  = 2 * sizeof(float);
    // Flat iterative integrator (ADR 0007): every TraceRay — primary, bounce,
    // shadow — launches from raygen; nothing traces from hit/miss shaders.
    desc.maxRecursionDepth = 1;
    return desc;
}

REGISTER_RAYTRACE_TECHNIQUE("Path Tracing", PathTracingPass)
