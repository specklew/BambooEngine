#pragma once
#include "DxrTechnique.h"

// Default path tracing technique — Cook-Torrance BRDF, shadow rays, multi-bounce.
// Overrides only GetTechniqueDesc(); everything else is inherited from DxrTechnique.
class PathTracingPass : public DxrTechnique
{
protected:
    TechniqueDesc GetTechniqueDesc() const override;
};
