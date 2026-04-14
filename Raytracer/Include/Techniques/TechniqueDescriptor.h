#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

enum class ShaderRole
{
    RayGen,
    Miss,
    ClosestHit,
    AnyHit,
};

struct ShaderDesc
{
    std::string  shaderPath;   // e.g. "resources/shaders/raytracing.rg.shader"
    std::wstring exportName;   // e.g. L"RayGen" — must match entry point in HLSL
    ShaderRole   role;
};

struct HitGroupDesc
{
    std::wstring name;
    std::wstring closestHitExport;   // export name of ClosestHit shader; empty if none
    std::wstring anyHitExport;       // export name of AnyHit shader; empty if none
};

struct TechniqueDesc
{
    std::vector<ShaderDesc>   shaders;
    std::vector<HitGroupDesc> hitGroups;
    uint32_t maxPayloadSize    = 8 * sizeof(float);  // default: 8 floats (RGB + throughput + bounce + seed)
    uint32_t maxAttributeSize  = 2 * sizeof(float);  // default: 2 floats (barycentric coords)
    uint32_t maxRecursionDepth = 8;
};

class RaytracePass;

struct TechniqueEntry
{
    std::string name;
    std::function<std::shared_ptr<RaytracePass>()> create;
};

// Place at file scope in a .cpp alongside the subclass definition.
// Runs before main() to self-register into RaytracePass::GetRegistry().
#define REGISTER_RAYTRACE_TECHNIQUE(Name, Class) \
    static int _reg_##Class = RaytracePass::RegisterTechnique( \
        Name, []() { return std::make_shared<Class>(); });
