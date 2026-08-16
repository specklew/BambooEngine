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

void RenderGraphPassBuilder::Declare(const BindingSlot& slot, GraphResourceHandle resource)
{
    // A slot the graph never sees still gets bound — it just has no producer or
    // consumer to order against, so declaring it would only add a false edge.
    assert(slot.graphAccess != GraphAccess::None &&
           "BindingSlot passed to Declare() has no graph access; wrap its slot in Accesses(...)");

    if (IsWriteAccess(slot.graphAccess))
        Write(resource, slot.graphAccess);
    else
        Read(resource, slot.graphAccess);
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
    case GraphAccess::None:
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
    case GraphAccess::None:                return "None";
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

void RenderGraph::AddPass(const char* name, const std::function<void(RenderGraphPassBuilder&)>& declare,
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
    pass.disabled     = m_disabledPasses.count(pass.name) != 0;
    m_passes.push_back(std::move(pass));
}

std::vector<RenderGraph::PassInfo> RenderGraph::GetPassInfo() const
{
    std::vector<PassInfo> info;
    info.reserve(m_passes.size());
    for (const PassNode& pass : m_passes)
        info.push_back({pass.name, pass.culled, pass.disabled});
    return info;
}

void RenderGraph::SetPassEnabled(const std::string& name, bool enabled)
{
    if (enabled)
        m_disabledPasses.erase(name);
    else
        m_disabledPasses.insert(name);
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

        // A disabled node is culled outright, and because its reads never mark
        // anything needed, whatever fed only it is culled with it.
        if (pass.disabled)
        {
            pass.culled = true;
            continue;
        }

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

// Two passes (ADR 0017 phase 6b). The first fixes the frame's node order, so the
// second knows the whole timeline and can place a transition anywhere between the
// node that last touched the resource and the node that needs the new state,
// instead of only immediately before the consumer.
void RenderGraph::Compile()
{
    m_compiled.clear();
    m_barrierLog.clear();

    Cull();

    // Pass one: the surviving nodes, in submission order. Barriers land in these
    // slots by index, which is what lets pass two write backwards.
    for (uint32_t index = 0; index < m_passes.size(); ++index)
    {
        if (m_passes[index].culled)
            continue;

        CompiledPass compiled;
        compiled.passIndex = index;
        compiled.queue     = m_passes[index].queue;
        m_compiled.push_back(std::move(compiled));
    }

    // Pass two: synthesize and place. Declarations are still walked in submission
    // order, so the phase-0 tracker sees exactly the sequence it always did.
    std::vector<uint32_t> lastAccessSlot(m_resources.size(), kNoSlot);

    for (uint32_t slot = 0; slot < m_compiled.size(); ++slot)
    {
        PassNode& pass = m_passes[m_compiled[slot].passIndex];

        for (const auto& declaration : pass.declarations)
        {
            assert(declaration.resource < m_resources.size() && "Declared resource was never imported");
            ImportedResource& imported = m_resources[declaration.resource];
            const D3D12_RESOURCE_STATES required = ToResourceState(declaration.access);

            // Earliest legal point: the slot after the previous access. Clamped to
            // the consumer, which covers both a resource declared twice by one node
            // and a producer that is the immediately preceding node. kNoSlot means
            // the previous access was in an earlier frame, so the top of this one is
            // legal.
            const uint32_t previous = lastAccessSlot[declaration.resource];
            const uint32_t earliest = (previous == kNoSlot) ? 0u : std::min(previous + 1, slot);

            bool emittedTransition = false;
            if (imported.tracked)
            {
                // The phase-0 tracker is the single source of truth for where the
                // resource currently is; the graph only names the destination.
                const D3D12_RESOURCE_STATES before = imported.tracked->GetTrackedState().state;
                if (before != required)
                {
                    const D3D12_RESOURCE_BARRIER barrier =
                        ResourceStateTracker::Get().BuildTransitionChecked(*imported.tracked, before, required);
                    const uint32_t placed = PlaceTransition(barrier, imported.raw, earliest, slot);
                    emittedTransition = true;

                    if (m_logBarriers)
                        m_barrierLog.push_back(pass.name + ": " + imported.debugName + " " +
                                               ResourceStateConversion::ToString(before) + " -> " +
                                               ResourceStateConversion::ToString(required) +
                                               DescribePlacement(placed, slot));
                }
            }
            else
            {
                assert(required == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && "Raw import declared with a state-carrying access — import it as a tracked Resource");
            }

            ID3D12Resource* underlying = imported.tracked
                ? imported.tracked->GetUnderlyingResource().Get()
                : imported.raw;

            // Hazards a transition cannot express. Read-after-write and
            // write-after-write are the obvious ones; write-after-read needs the
            // barrier too, or the writer can overwrite while readers are still in
            // flight (the injection-clear vs guiding-build edge). A transition
            // already orders both sides, so it stands in for the UAV barrier.
            //
            // A never-seen resource defaults to neither, so its first declaration
            // emits nothing — there is no producer to order against.
            //
            // Placed at the consumer whatever the mode: a UAV barrier changes no
            // state, so there is nothing for an earlier placement to overlap with —
            // it would only move the drain in front of the nodes in between.
            UavUsage&  usage      = m_uavUsage[underlying];
            const bool afterWrite = usage.writtenSinceLastRead;
            const bool afterRead  = usage.readSinceLastWrite && declaration.isWrite;
            if (!emittedTransition && required == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && (afterWrite || afterRead))
            {
                m_compiled[slot].barriers.push_back({CD3DX12_RESOURCE_BARRIER::UAV(underlying), underlying});

                if (m_logBarriers)
                    m_barrierLog.push_back(pass.name + ": UAV " + imported.debugName +
                                           (usage.lastTouchedFrame == m_frameCounter ? "" : " (cross-frame)"));
            }

            usage.writtenSinceLastRead = declaration.isWrite;
            usage.readSinceLastWrite   = !declaration.isWrite;
            usage.lastTouchedFrame     = m_frameCounter;

            lastAccessSlot[declaration.resource] = slot;
        }
    }
}

// Returns the slot the barrier (or its BEGIN half) went to, for the log.
uint32_t RenderGraph::PlaceTransition(const D3D12_RESOURCE_BARRIER& barrier, ID3D12Resource* resource,
                                      uint32_t earliestSlot, uint32_t consumerSlot)
{
    // Nothing to gain from a split or a hoist when the producer is the node right
    // before the consumer — there is no work in between to overlap with.
    if (m_barrierPlacement == BarrierPlacement::Consumer || earliestSlot == consumerSlot)
    {
        m_compiled[consumerSlot].barriers.push_back({barrier, resource});
        return consumerSlot;
    }

    if (m_barrierPlacement == BarrierPlacement::Earliest)
    {
        m_compiled[earliestSlot].barriers.push_back({barrier, resource});
        return earliestSlot;
    }

    // Split. The resource must not be touched between the two halves, which the
    // earliest-legal interval guarantees: it is bounded by its own producer and
    // consumer. Both halves are recorded into the one command list the graph
    // executes into, so the pair can never straddle an ExecuteCommandLists.
    D3D12_RESOURCE_BARRIER begin = barrier;
    begin.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
    D3D12_RESOURCE_BARRIER end = barrier;
    end.Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;

    m_compiled[earliestSlot].barriers.push_back({begin, resource});
    m_compiled[consumerSlot].barriers.push_back({end, resource});
    return earliestSlot;
}

std::string RenderGraph::DescribePlacement(uint32_t placedSlot, uint32_t consumerSlot) const
{
    if (placedSlot == consumerSlot)
        return {};

    const std::string producer = m_passes[m_compiled[placedSlot].passIndex].name;
    const std::string span     = std::to_string(consumerSlot - placedSlot);
    return (m_barrierPlacement == BarrierPlacement::Split)
        ? " [split from '" + producer + "', " + span + " nodes]"
        : " [hoisted to '" + producer + "', " + span + " nodes early]";
}

void RenderGraph::Execute(CommandContext& context)
{
    m_timedPassCount = 0;
    m_timedPassNames.clear();

    for (const CompiledPass& compiled : m_compiled)
    {
        PassNode& pass = m_passes[compiled.passIndex];

        for (const GraphBarrier& barrier : compiled.barriers)
            context.EnqueueBarrier(barrier.barrier, barrier.resource);

        // PIX event per node, named from the declaration — no opt-in, every pass.
        ScopedGpuMarker marker(context.GetCommandListUnflushed(), pass.name.c_str());

        // The opening timestamp goes after the barriers so a node is charged for
        // its own work, not for the wait its predecessor made necessary.
        const bool timed = m_timingEnabled && m_timedPassCount < kMaxTimedPasses;
        const uint32_t timerSlot = m_timedPassCount * 2;
        if (timed)
        {
            context.FlushBarriers();
            context.GetCommandListUnflushed()->EndQuery(m_timerQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timerSlot);
        }

        if (pass.execute)
            pass.execute();
        else
            context.FlushBarriers(); // transition-only node: nothing to flush it

        if (timed)
        {
            context.FlushBarriers();
            context.GetCommandListUnflushed()->EndQuery(m_timerQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timerSlot + 1);
            m_timedPassNames.push_back(pass.name);
            ++m_timedPassCount;
        }
    }

    context.FlushBarriers();

    if (m_timedPassCount > 0)
    {
        context.GetCommandListUnflushed()->ResolveQueryData(m_timerQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
            m_timedPassCount * 2, m_timerReadback.Get(), 0);
    }
}

void RenderGraph::InitializeTimers(ID3D12Device* device, ID3D12CommandQueue* queue)
{
    if (!device || !queue || FAILED(queue->GetTimestampFrequency(&m_timerFrequency)))
        return;

    D3D12_QUERY_HEAP_DESC heapDesc = {};
    heapDesc.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heapDesc.Count = kMaxTimedPasses * 2;
    if (FAILED(device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&m_timerQueryHeap))))
        return;
    m_timerQueryHeap->SetName(L"RenderGraph Timestamps");

    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    const auto bufferDesc     = CD3DX12_RESOURCE_DESC::Buffer(heapDesc.Count * sizeof(uint64_t));
    if (FAILED(device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_timerReadback))))
    {
        m_timerQueryHeap.Reset();
        return;
    }
    m_timerReadback->SetName(L"RenderGraph Timestamp Readback");
}

void RenderGraph::ResolveTimings()
{
    if (m_timedPassCount == 0 || !m_timerReadback || m_timerFrequency == 0)
        return;

    const D3D12_RANGE readRange{0, m_timedPassCount * 2 * sizeof(uint64_t)};
    uint64_t* ticks = nullptr;
    if (FAILED(m_timerReadback->Map(0, &readRange, reinterpret_cast<void**>(&ticks))))
        return;

    m_timings.clear();
    const double msPerTick = 1000.0 / static_cast<double>(m_timerFrequency);
    for (uint32_t i = 0; i < m_timedPassCount; ++i)
    {
        const uint64_t begin = ticks[i * 2];
        const uint64_t end   = ticks[i * 2 + 1];
        const uint64_t span  = (end > begin) ? end - begin : 0;
        const float milliseconds = static_cast<float>(span * msPerTick);
        m_timings.push_back({m_timedPassNames[i], milliseconds});

        TimingAccumulator& accumulator = m_timingHistory[m_timedPassNames[i]];
        accumulator.totalMilliseconds += milliseconds;
        accumulator.maxMilliseconds = std::max(accumulator.maxMilliseconds, milliseconds);
        ++accumulator.frames;
    }

    const D3D12_RANGE writeRange{0, 0};
    m_timerReadback->Unmap(0, &writeRange);
}

std::vector<RenderGraph::PassTimingSummary> RenderGraph::GetTimingSummary() const
{
    std::vector<PassTimingSummary> summary;
    summary.reserve(m_timingHistory.size());
    for (const auto& [name, accumulator] : m_timingHistory)
    {
        const float mean = accumulator.frames > 0
            ? static_cast<float>(accumulator.totalMilliseconds / accumulator.frames)
            : 0.0f;
        summary.push_back({name, mean, accumulator.maxMilliseconds, accumulator.frames});
    }

    std::sort(summary.begin(), summary.end(),
        [](const PassTimingSummary& a, const PassTimingSummary& b) { return a.meanMilliseconds > b.meanMilliseconds; });
    return summary;
}

void RenderGraph::Reset()
{
    m_passes.clear();
    m_compiled.clear();
    m_resources.clear();
    ++m_frameCounter;

    for (auto entry = m_uavUsage.begin(); entry != m_uavUsage.end();)
        entry = (entry->second.lastTouchedFrame + kUavUsageLifetimeFrames < m_frameCounter)
            ? m_uavUsage.erase(entry)
            : std::next(entry);
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
    // The mode is part of the reading: the same frame produces a different list
    // under each placement, so a dump that does not name it cannot be compared.
    std::string dump = "placement: ";
    switch (m_barrierPlacement)
    {
    case BarrierPlacement::Consumer: dump += "at consumer\n";   break;
    case BarrierPlacement::Earliest: dump += "earliest legal\n"; break;
    case BarrierPlacement::Split:    dump += "split begin/end\n"; break;
    }

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
