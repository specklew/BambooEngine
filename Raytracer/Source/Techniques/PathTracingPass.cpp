#include "pch.h"
#include "Techniques/PathTracingPass.h"

TechniqueDesc PathTracingPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
            {"resources/shaders/raytracing.rg.shader",          L"RayGen",     ShaderRole::RayGen},
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
    desc.maxPayloadSize    = 8 * sizeof(float);
    desc.maxAttributeSize  = 2 * sizeof(float);
    desc.maxRecursionDepth = 8;
    return desc;
}

REGISTER_RAYTRACE_TECHNIQUE("Path Tracing", PathTracingPass)
