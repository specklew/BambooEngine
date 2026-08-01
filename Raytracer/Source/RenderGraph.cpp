#include "pch.h"
#include "RenderGraph.h"

#include "CommandContext.h"
#include "Resources/Resource.h"
#include "Resources/ResourceStateTracker.h"

void RenderGraphPassBuilder::Read(GraphResourceHandle resource, GraphAccess access)
{
    m_declarations.push_back({resource, access, false});
}

void RenderGraphPassBuilder::Write(GraphResourceHandle resource, GraphAccess access)
{
    m_declarations.push_back({resource, access, true});
}

D3D12_RESOURCE_STATES RenderGraph::ToResourceState(GraphAccess access)
{
    switch (access)
    {
    case GraphAccess::ComputeRead:        return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case GraphAccess::ComputeWrite:
    case GraphAccess::UnorderedAccessRead: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case GraphAccess::PixelRead:        return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    case GraphAccess::RenderTarget:     return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case GraphAccess::DepthWrite:       return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case GraphAccess::IndirectArgument: return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    case GraphAccess::CopySource:       return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case GraphAccess::CopyDestination:  return D3D12_RESOURCE_STATE_COPY_DEST;
    case GraphAccess::Present:          return D3D12_RESOURCE_STATE_PRESENT;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

GraphResourceHandle RenderGraph::Import(Resource& resource, const char* debugName)
{
    for (size_t i = 0; i < m_resources.size(); ++i)
        if (m_resources[i].tracked == &resource)
            return static_cast<GraphResourceHandle>(i);

    ImportedResource imported;
    imported.tracked   = &resource;
    imported.raw       = resource.GetUnderlyingResource().Get();
    imported.debugName = debugName ? debugName : "<unnamed>";
    m_resources.push_back(std::move(imported));
    return static_cast<GraphResourceHandle>(m_resources.size() - 1);
}

GraphResourceHandle RenderGraph::ImportRaw(ID3D12Resource* resource, const char* debugName)
{
    for (size_t i = 0; i < m_resources.size(); ++i)
        if (m_resources[i].raw == resource)
            return static_cast<GraphResourceHandle>(i);

    ImportedResource imported;
    imported.raw       = resource;
    imported.debugName = debugName ? debugName : "<unnamed>";
    m_resources.push_back(std::move(imported));
    return static_cast<GraphResourceHandle>(m_resources.size() - 1);
}

void RenderGraph::AddPass(const char* name,
                          const std::function<void(RenderGraphPassBuilder&)>& declare,
                          std::function<void()> execute)
{
    RenderGraphPassBuilder builder;
    declare(builder);

    PassNode pass;
    pass.m_name       = name ? name : "<unnamed pass>";
    pass.declarations = std::move(builder.m_declarations);
    pass.execute      = std::move(execute);
    m_passes.push_back(std::move(pass));
}

void RenderGraph::Execute(CommandContext& context)
{
    m_barrierLog.clear();

    for (PassNode& pass : m_passes)
    {
        for (const auto& declaration : pass.declarations)
        {
            assert(declaration.resource < m_resources.size() && "Declared resource was never imported");
            ImportedResource& imported = m_resources[declaration.resource];
            const D3D12_RESOURCE_STATES required = ToResourceState(declaration.access);

            // Hazards a transition cannot express. Read-after-write and
            // write-after-write are the obvious ones; write-after-read needs the
            // barrier too, or the writer can overwrite while readers are still in
            // flight (the injection-clear vs guiding-build edge).
            const bool afterWrite = imported.writtenSinceLastRead;
            const bool afterRead  = imported.readSinceLastWrite && declaration.isWrite;
            const bool needsUavBarrier = required == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
                                         imported.hasStateInGraph &&
                                         (afterWrite || afterRead);

            if (needsUavBarrier)
            {
                if (imported.tracked)
                    context.UavBarrier(*imported.tracked);
                else
                    context.UavBarrierRaw(imported.raw);

                m_barrierLog.push_back(pass.m_name + ": UAV " + imported.debugName);
            }
            else if (imported.tracked && (!imported.hasStateInGraph || imported.stateInGraph != required))
            {
                // Tracked resources know their own state, so the graph asks for
                // the destination and the phase-0 tracker validates the source.
                const D3D12_RESOURCE_STATES before = imported.tracked->GetTrackedState().state;
                if (before != required)
                {
                    context.Transition(*imported.tracked, before, required);
                    m_barrierLog.push_back(pass.m_name + ": " + imported.debugName + " " +
                                           ResourceStateConversion::ToString(before) + " -> " +
                                           ResourceStateConversion::ToString(required));
                }
            }

            imported.stateInGraph         = required;
            imported.hasStateInGraph      = true;
            imported.writtenSinceLastRead = declaration.isWrite;
            imported.readSinceLastWrite   = !declaration.isWrite;
        }

        if (pass.execute)
            pass.execute();
    }

    context.FlushBarriers();
}

void RenderGraph::Reset()
{
    m_passes.clear();
    for (ImportedResource& imported : m_resources)
    {
        imported.hasStateInGraph      = false;
        imported.writtenSinceLastRead = false;
    }
}

std::string RenderGraph::DumpPasses() const
{
    static const char* accessNames[] = {
        "ComputeRead", "ComputeWrite", "UnorderedAccessRead", "PixelRead",
        "RenderTarget", "DepthWrite", "IndirectArgument", "CopySource",
        "CopyDestination", "Present"
    };

    std::string dump;
    for (const PassNode& pass : m_passes)
    {
        dump += pass.m_name;
        dump += '\n';
        for (const auto& declaration : pass.declarations)
        {
            dump += "    ";
            dump += declaration.isWrite ? "writes " : "reads  ";
            dump += m_resources[declaration.resource].debugName;
            dump += " (";
            dump += accessNames[static_cast<uint32_t>(declaration.access)];
            dump += ")\n";
        }
    }
    return dump;
}

std::string RenderGraph::DumpBarriers() const
{
    std::string dump;
    for (const std::string& entry : m_barrierLog)
    {
        dump += entry;
        dump += '\n';
    }
    return dump;
}
