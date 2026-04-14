#pragma once
#include "RaytracePass.h"

// Default path tracing technique — Cook-Torrance BRDF, shadow rays, multi-bounce.
// Overrides only GetTechniqueDesc(); everything else is inherited from RaytracePass.
class PathTracingPass : public RaytracePass
{
protected:
    TechniqueDesc GetTechniqueDesc() const override;
};
