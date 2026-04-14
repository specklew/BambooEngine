#include "pch.h"
#include "Techniques/AmbientOcclusionPass.h"

TechniqueDesc AmbientOcclusionPass::GetTechniqueDesc() const
{
    TechniqueDesc desc;
    desc.shaders = {
        {"resources/shaders/ao.rg.shader",                 L"RayGen",    ShaderRole::RayGen},
        {"resources/shaders/ao.ms.shader",                 L"Miss",      ShaderRole::Miss},
        {"resources/shaders/ao.ch.shader",                 L"Hit",       ShaderRole::ClosestHit},
        {"resources/shaders/raytracing.shadowhit.shader",  L"ShadowHit", ShaderRole::AnyHit},
        {"resources/shaders/raytracing.shadowmiss.shader", L"ShadowMiss",ShaderRole::Miss},
    };
    desc.hitGroups = {
        {L"PrimaryHitGroup", L"Hit", L""},
        {L"ProbeHitGroup",   L"",    L"ShadowHit"},
    };
    desc.maxPayloadSize    = 8 * sizeof(float); // reuses Payload struct (32 bytes)
    desc.maxAttributeSize  = 2 * sizeof(float);
    desc.maxRecursionDepth = 1; // no recursive bouncing; all TraceRay calls from RayGen
    return desc;
}

REGISTER_RAYTRACE_TECHNIQUE("Ambient Occlusion", AmbientOcclusionPass)
