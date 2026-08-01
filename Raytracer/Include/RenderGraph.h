#pragma once
#include <functional>
#include <string>
#include <vector>

class CommandContext;
class Resource;

// ADR 0017 L5, step A: passes declare what they read and write; the graph culls
// the passes nothing consumes, turns the surviving declarations into barriers,
// and runs them in dependency order. Bindings are untouched at this step — a
// converted pass records exactly what it recorded before, minus its hand-placed
// barriers.
//
// Compile() and Execute() are separate on purpose: Compile() produces a barrier
// plan as data, Execute() submits it. Async compute (phase 6) inserts a
// scheduler between the two rather than rewriting the compiler.

using GraphResourceHandle = uint32_t;
inline constexpr GraphResourceHandle InvalidGraphResource = ~0u;

// What a pass does to a resource. The graph maps these to the legacy states the
// barrier needs; a Write followed by a Read is the edge that orders two passes.
enum class GraphAccess
{
    ComputeRead,          // NON_PIXEL_SHADER_RESOURCE
    ComputeWrite,         // UNORDERED_ACCESS
    UnorderedAccessRead,  // read through a UAV binding — same state, still needs
                          // a UAV barrier after a writer (the VXPG textures)
    PixelRead,        // PIXEL_SHADER_RESOURCE
    RenderTarget,
    DepthWrite,
    IndirectArgument,
    CopySource,
    CopyDestination,
    Present,
    Count
};

// Declared now, one legal value until phase 6. Cross-queue synchronisation is
// fences rather than barriers and constrains which states a node may ask for, so
// the compiler carries the attribute from the start instead of baking in the
// single-queue assumption (ADR 0017).
enum class GraphQueue
{
    Direct,
    AsyncCompute,
    Copy
};

class RenderGraphPassBuilder
{
public:
    void Read(GraphResourceHandle resource, GraphAccess access);
    void Write(GraphResourceHandle resource, GraphAccess access);

    void SetQueue(GraphQueue queue) { m_queue = queue; }

    // Survives culling regardless of who reads its outputs: for side effects the
    // declarations cannot express (a transition-only node, a producer whose
    // consumer is the next frame, a sink that reads without writing).
    void NeverCull() { m_neverCull = true; }

private:
    friend class RenderGraph;

    struct Declaration
    {
        GraphResourceHandle resource;
        GraphAccess         access;
        bool                isWrite;
    };

    std::vector<Declaration> m_declarations;
    GraphQueue               m_queue     = GraphQueue::Direct;
    bool                     m_neverCull = false;
};

class RenderGraph
{
public:
    // Resources are imported, not owned: transient allocation and aliasing are
    // phase 6. A tracked Resource keeps its phase-0 state model; raw ComPtr
    // resources (the VXPG textures) carry no state and only get UAV barriers.
    // Imports last one frame — Reset() drops them, so a resource destroyed by a
    // resize or a scene switch can never be reached through a stale pointer.
    GraphResourceHandle Import(Resource& resource, const char* debugName);
    GraphResourceHandle ImportRaw(ID3D12Resource* resource, const char* debugName);

    // Consumed outside the graph (the presented back buffer, a resource the
    // not-yet-converted raster draws sample). Keeps its producers alive.
    void MarkExternallyRead(GraphResourceHandle resource);

    void AddPass(const char* name,
                 const std::function<void(RenderGraphPassBuilder&)>& declare,
                 std::function<void()> execute);

    // Culls, then synthesizes the barrier plan. Advances the phase-0 tracker, so
    // every Compile() must be followed by exactly one Execute().
    void Compile();

    // Submits the compiled plan: barriers, PIX event, pass body, in that order.
    void Execute(CommandContext& context);

    // Drops passes and imports; the next frame declares itself from scratch.
    void Reset();

    // Off by default — building the strings costs an allocation per barrier.
    void SetBarrierLogging(bool enabled) { m_logBarriers = enabled; }

    // Per-pass GPU timing. Two timestamps per surviving node plus one
    // ResolveQueryData per frame, so it stays opt-in: benchmark runs must not pay
    // for it. Call ResolveTimings() once the GPU has finished the frame.
    void InitializeTimers(ID3D12Device* device, ID3D12CommandQueue* queue);
    void SetTimingEnabled(bool enabled) { m_timingEnabled = enabled && m_timerQueryHeap != nullptr; }
    void ResolveTimings();

    struct PassTiming
    {
        std::string name;
        float       milliseconds;
    };
    [[nodiscard]] const std::vector<PassTiming>& GetTimings() const { return m_timings; }

    // Barrier attribution for a perf delta (ADR 0017: this exists to explain a
    // regression, not to gate on byte-identical output).
    [[nodiscard]] std::string DumpBarriers() const;

    // Node list with each node's declarations, whether or not it emitted a
    // barrier — the "what does the frame actually consist of" view. Culled
    // passes are listed and marked.
    [[nodiscard]] std::string DumpPasses() const;

    // Mermaid flowchart of the compiled graph: nodes, resource edges, culled
    // passes. Text, so it renders in VS Code and GitHub with no dependency.
    [[nodiscard]] std::string DumpMermaid() const;

private:
    struct ImportedResource
    {
        Resource*       tracked = nullptr; // null for raw imports
        ID3D12Resource* raw     = nullptr;
        std::string     debugName;
        bool            externallyRead = false;
        // UAV hazard bookkeeping, valid only within one compiled frame.
        bool            touchedThisFrame     = false;
        bool            writtenSinceLastRead = false;
        bool            readSinceLastWrite   = false;
    };

    struct PassNode
    {
        std::string                                      name;
        std::vector<RenderGraphPassBuilder::Declaration> declarations;
        std::function<void()>                            execute;
        GraphQueue                                       queue     = GraphQueue::Direct;
        bool                                             neverCull = false;
        bool                                             culled    = false;
    };

    struct GraphBarrier
    {
        D3D12_RESOURCE_BARRIER barrier;
        ID3D12Resource*        resource; // batching identity: two barriers on one
                                         // resource must not share a call
    };

    // Compiler output is data, not action.
    struct CompiledPass
    {
        uint32_t                  passIndex;
        GraphQueue                queue;
        std::vector<GraphBarrier> barriers;
    };

    static D3D12_RESOURCE_STATES ToResourceState(GraphAccess access);
    static const char*           ToString(GraphAccess access);

    void Cull();

    // Two timestamps per node; the cap is the node budget one frame may time.
    static constexpr uint32_t kMaxTimedPasses = 64;

    std::vector<ImportedResource> m_resources;
    std::vector<PassNode>         m_passes;
    std::vector<CompiledPass>     m_compiled;
    std::vector<std::string>      m_barrierLog;
    bool                          m_logBarriers = false;

    Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_timerQueryHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource>  m_timerReadback;
    uint64_t                                m_timerFrequency = 0;
    bool                                    m_timingEnabled  = false;
    uint32_t                                m_timedPassCount = 0;
    std::vector<std::string>                m_timedPassNames;
    std::vector<PassTiming>                 m_timings;
};
