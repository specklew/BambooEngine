#pragma once
#include "RaytracePass.h"

class AmbientOcclusionPass : public RaytracePass
{
protected:
    TechniqueDesc GetTechniqueDesc() const override;
};
