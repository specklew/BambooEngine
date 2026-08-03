#pragma once
#include "DxrTechnique.h"

class AmbientOcclusionPass : public DxrTechnique
{
protected:
    TechniqueDesc GetTechniqueDesc() const override;
};
