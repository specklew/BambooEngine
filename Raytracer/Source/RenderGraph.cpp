#include "pch.h"
#include "RenderGraph.h"

#include "CommandContext.h"
#include "Resources/Resource.h"
#include "Resources/ResourceStateTracker.h"
#include "Utils/GpuMarker.h"

void RenderGraphPassBuilder::Read(GraphResourceHandle resource, GraphAccess access)
{
    if (resource != InvalidGraphResource)
        m_declarations.push_back({resource, access, false});
}

void RenderGraphPassBuilder::Write(GraphResourceHandle resource, GraphAccess access)
{
    if (resource != InvalidGraphResource)
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
    case GraphAccess::Count:            break;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

const char* RenderGraph::ToString(GraphAccess access)
{
    switch (access)
    {
    case GraphAccess::ComputeRead:         return "ComputeRead";
    case GraphAccess::ComputeWrite:        return "ComputeWrite";
    case GraphAccess::UnorderedAccessRead: return "UnorderedAccessRead";
    case GraphAccess::PixelRead:           return "PixelRead";
    case GraphAccess::RenderTarget:        return "RenderTarget";
    case GraphAccess::DepthWrite:          return "DepthWrite";
    case GraphAccess::IndirectArgument:    return "IndirectArgument";
    case GraphAccess::CopySource:          return "CopySource";
    case GraphAccess::CopyDestination:     return "CopyDestination";
    case GraphAccess::Present:             return "Present";
    case GraphAccess::Count:               break;
    }
    return "<unknown>";
}

GraphResourceHandle RenderGraph::Import(Resource& resource, const char* debugName)
{
    ID3D12Resource* underlying = resource.GetUnderlyingResource().Get();

    for (size_t i = 0; i < m_resources.size(); ++i)
    {
        if (m_resources[i].tracked == &resource || m_resources[i].raw == underlying)
        {
            // A raw import may have claimed this resource first; the tracked view
            // is the better one (it can carry transitions), so upgrade in place.
            m_resources[i].tracked = &resource;
            return static_cast<GraphResourceHandle>(i);
        }
    }

    ImportedResource imported;
    imported.tracked   = &resource;
    imported.raw       = underlying;
    imported.debugName = debugName ? debugName : "<unnamed>";
    m_resources.push_back(std::move(imported));
    return static_cast<GraphResourceHandle>(m_resources.size() - 1);
}

GraphResourceHandle RenderGraph::ImportRaw(ID3D12Resource* resource, const char* debugName)
{
    if (!resource)
        return InvalidGraphResource;

    for (size_t i = 0; i < m_resources.size(); ++i)
        if (m_resources[i].raw == resource)
            return static_cast<GraphResourceHandle>(i);

    ImportedResource imported;
    imported.raw       = resource;
    imported.debugName = debugName ? debugName : "<unnamed>";
    m_resources.push_back(std::move(imported));
    return static_cast<GraphResourceHandle>(m_resources.size() - 1);
}

void RenderGraph::MarkExternallyRead(GraphResourceHandle resource)
{
    if (resource != InvalidGraphResource)
        m_resources[resource].externallyRead = true;
}

void RenderGraph::AddPass(const char* name,
                          const std::function<void(RenderGraphPassBuilder&)>& declare,
                          std::function<void()> execute)
{
    RenderGraphPassBuilder builder;
    declare(builder);

    PassNode pass;
    pass.name         = name ? name : "<unnamed pass>";
    pass.declarations = std::move(builder.m_declarations);
    pass.execute      = std::move(execute);
    pass.queue        = builder.m_queue;
    pass.neverCull    = builder.m_neverCull;
    m_passes.push_back(std::move(pass));
}

// Backward reachability over the declaration order: a pass survives if it is a
// sink, or if it writes something a surviving pass (or the world outside the
// graph) still needs. This replaces the hand-maintained VxpgStage ladder — the
// graph derives which stages must run from the dependency data it already needs
// for barriers.
void RenderGraph::Cull()
{
    std::vector<bool> needed(m_resources.size(), false);
    for (size_t i = 0; i < m_resources.size(); ++i)
        needed[i] = m_resources[i].externallyRead;

    for (size_t index = m_passes.size(); index-- > 0;)
    {
        PassNode& pass = m_passes[index];

        bool alive = pass.neverCull;
        if (!alive)
        {
            for (const auto& declaration : pass.declarations)
                if (declaration.isWrite && needed[declaration.resource])
                {
                    alive = true;
                    break;
                }
        }

        pass.culled = !alive;
        if (!alive)
            continue;

        // A write is not assumed to fully overwrite, so the resource stays needed
        // for earlier producers too — conservative, and correct for partial writes.
        for (const auto& declaration : pass.declarations)
            if (!declaration.isWrite)
                needed[declaration.resource] = true;
    }
}

void RenderGraph::Compile()
{
    m_compiled.clear();
    m_barrierLog.clear();

    Cull();

    for (uint32_t index = 0; index < m_passes.size(); ++index)
    {
        PassNode& pass = m_passes[index];
        if (pass.culled)
            continue;

        CompiledPass compiled;
        compiled.passIndex = index;
        compiled.queue     = pass.queue;

        for (const auto& declaration : pass.declarations)
        {
            assert(declaration.resource < m_resources.size() && "Declared resource was never imported");
            ImportedResource& imported = m_resources[declaration.resource];
            const D3D12_RESOURCE_STATES required = ToResourceState(declaration.access);

            bool emittedTransition = false;
            if (imported.tracked)
            {
                // The phase-0 tracker is the single source of truth for where the
                // resource currently is; the graph only names the destination.
                const D3D12_RESOURCE_STATES before = imported.tracked->GetTrackedState().state;
                if (before != required)
                {
                    compiled.barriers.push_back({
                        ResourceStateTracker::Get().BuildTransitionChecked(*imported.tracked, before, required),
                        imported.raw});
                    emittedTransition = true;

                    if (m_logBarriers)
                        m_barrierLog.push_back(pass.name + ": " + imported.debugName + " " +
                                               ResourceStateConversion::ToString(before) + " -> " +
                                               ResourceStateConversion::ToString(required));
                }
            }
            else
            {
                assert(required == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
                       "Raw import declared with a state-carrying access — import it as a tracked Resource");
            }

            // Hazards a transition cannot express. Read-after-write and
            // write-after-write are the obvious ones; write-after-read needs the
            // barrier too, or the writer can overwrite while readers are still in
            // flight (the injection-clear vs guiding-build edge). A transition
            // already orders both sides, so it stands in for the UAV barrier.
            const bool afterWrite = imported.writtenSinceLastRead;
            const bool afterRead  = imported.readSinceLastWrite && declaration.isWrite;
            if (!emittedTransition && required == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
                imported.touchedThisFrame && (afterWrite || afterRead))
            {
                ID3D12Resource* underlying = imported.tracked
                    ? imported.tracked->GetUnderlyingResource().Get()
                    : imported.raw;
                compiled.barriers.push_back({CD3DX12_RESOURCE_BARRIER::UAV(underlying), underlying});

                if (m_logBarriers)
                    m_barrierLog.push_back(pass.name + ": UAV " + imported.debugName);
            }

            imported.touchedThisFrame     = true;
            imported.writtenSinceLastRead = declaration.isWrite;
            imported.readSinceLastWrite   = !declaration.isWrite;
        }

        m_compiled.push_back(std::move(compiled));
    }
}

void RenderGraph::Execute(CommandContext& context)
{
    for (const CompiledPass& compiled : m_compiled)
    {
        PassNode& pass = m_passes[compiled.passIndex];

        for (const GraphBarrier& barrier : compiled.barriers)
            context.EnqueueBarrier(barrier.barrier, barrier.resource);

        // PIX event per node, named from the declaration — no opt-in, every pass.
        ScopedGpuMarker marker(context.GetCommandListUnflushed(), pass.name.c_str());

        if (pass.execute)
            pass.execute();
        else
            context.FlushBarriers(); // transition-only node: nothing to flush it
    }

    context.FlushBarriers();
}

void RenderGraph::Reset()
{
    m_passes.clear();
    m_compiled.clear();
    m_resources.clear();
}

std::string RenderGraph::DumpPasses() const
{
    std::string dump;
    for (const PassNode& pass : m_passes)
    {
        dump += pass.culled ? "[culled] " : "         ";
        dump += pass.name;
        dump += '\n';
        for (const auto& declaration : pass.declarations)
        {
            dump += "    ";
            dump += declaration.isWrite ? "writes " : "reads  ";
            dump += m_resources[declaration.resource].debugName;
            dump += " (";
            dump += ToString(declaration.access);
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

std::string RenderGraph::DumpMermaid() const
{
    auto mermaidId = [](const std::string& name)
    {
        std::string id;
        for (const char c : name)
            id += (std::isalnum(static_cast<unsigned char>(c)) != 0) ? c : '_';
        return id;
    };

    std::string dump = "flowchart LR\n";

    for (const PassNode& pass : m_passes)
    {
        dump += "    " + mermaidId(pass.name) + "[\"" + pass.name + "\"]";
        dump += pass.culled ? ":::culled\n" : "\n";
    }

    // One edge per producer -> consumer pair, labelled with the resource.
    std::vector<std::string> lastWriter(m_resources.size());
    for (const PassNode& pass : m_passes)
    {
        if (pass.culled)
            continue;

        for (const auto& declaration : pass.declarations)
        {
            const std::string& producer = lastWriter[declaration.resource];
            if (!declaration.isWrite && !producer.empty())
                dump += "    " + mermaidId(producer) + " -->|" +
                        m_resources[declaration.resource].debugName + "| " + mermaidId(pass.name) + "\n";
        }
        for (const auto& declaration : pass.declarations)
            if (declaration.isWrite)
                lastWriter[declaration.resource] = pass.name;
    }

    dump += "    classDef culled stroke-dasharray: 4 4,color:#888\n";
    return dump;
}
