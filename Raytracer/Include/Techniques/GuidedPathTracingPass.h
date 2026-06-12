#pragma once

#include "RaytracePass.h"

// VXPG guided path tracing technique. Stage A: MIS between BSDF sampling and
// a uniform-sphere guide at the first bounce (validates MIS plumbing against
// vanilla PT). Stage B will swap the guide for the voxel irradiance CDF.
class GuidedPathTracingPass : public RaytracePass
{
protected:
    TechniqueDesc GetTechniqueDesc() const override;
};
