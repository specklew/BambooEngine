#include "pch.h"
#include "RenderTechnique.h"

std::vector<RenderTechnique::Entry>& RenderTechnique::GetRegistry()
{
    static std::vector<Entry> registry;
    return registry;
}

int RenderTechnique::RegisterTechnique(std::string name, std::function<std::shared_ptr<RenderTechnique>()> factory)
{
    GetRegistry().push_back({std::move(name), std::move(factory)});
    return static_cast<int>(GetRegistry().size()) - 1;
}
