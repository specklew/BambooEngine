#pragma once
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "BindingSlot.h"
#include "GraphAccess.h"

class CommandContext;
class GraphicsDevice;
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

class RenderGraphPassBuilder
{
public:
    void Read(GraphResourceHandle resource, GraphAccess access);
    void Write(GraphResourceHandle resource, GraphAccess access);

    // Declares a binding the pass already describes in its slot table: the slot
    // says what the access is, the node only supplies this frame's resource
    // (ADR 0017 step 3). One declaration, two consumers — the root signature and
    // the graph.
    void Declare(const BindingSlot& slot, GraphResourceHandle resource);

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

    void AddPass(const char* name, const std::function<void(RenderGraphPassBuilder&)>& declare,
                 std::function<void()> execute);

    // Where a synthesized transition is placed (ADR 0017 phase 6b). Compile()
    // knows the whole frame before it emits anything, so a transition is legal
    // anywhere between the node that last touched the resource and the node that
    // needs the new state — nothing in between may touch it, by construction.
    //
    //   Consumer — immediately before the node that needs it (phases 3-6a).
    //   Earliest — immediately after the previous access, so the drain the
    //              barrier forces happens ahead of the intervening nodes instead
    //              of behind them.
    //   Split    — BEGIN at the earliest legal point, END before the consumer, so
    //              the driver may overlap the transition with whatever runs in
    //              between. This is the placement the phase-6b lever is about;
    //              the other two exist to measure it against.
    //
    // Transitions only. A UAV barrier carries no state change to overlap, so
    // hoisting one only moves its drain earlier — it stays at its consumer.
    enum class BarrierPlacement
    {
        Consumer,
        Earliest,
        Split,
    };
    void SetBarrierPlacement(BarrierPlacement placement) { m_barrierPlacement = placement; }

    // Culls, then synthesizes the barrier plan. Advances the phase-0 tracker, so
    // every Compile() must be followed by exactly one Execute().
    void Compile();

    // Async compute (ADR 0017 phase 6c). Off means every node is forced onto the
    // direct queue and the frame is one submission, exactly as before phase 6c.
    void SetAsyncCompute(bool enabled) { m_asyncCompute = enabled; }

    // Records the compiled plan — barriers, PIX event, pass body per node — and
    // submits it. With one queue that is one command list and one
    // ExecuteCommandLists; with two it is a run per queue, each on its own list,
    // ordered by the cross-queue fences Compile() worked out.
    //
    // primaryDirectList is the frame's already-open direct list: it carries
    // whatever was recorded before the graph ran, so it has to be the first
    // submission rather than one the graph allocates.
    void ExecuteAndSubmit(CommandContext& context, GraphicsDevice& device,
                          ID3D12GraphicsCommandList4* primaryDirectList);

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

    // Running per-node statistics over every timed frame since the last reset. One
    // frame is far too noisy to rank nodes by — and the first frame is the worst
    // possible sample, since it pays for PSO creation and the geometry bake.
    struct PassTimingSummary
    {
        std::string name;
        float       meanMilliseconds;
        float       maxMilliseconds;
        uint32_t    frames;
    };
    [[nodiscard]] std::vector<PassTimingSummary> GetTimingSummary() const;
    void ResetTimingHistory() { m_timingHistory.clear(); }

    // Pass toggles. Disabling a node drops it and re-culls, so its producers go
    // too if nothing else reads them — the debug view becomes "a pass you don't
    // add" rather than a branch inside a shader. Consumers still run and read
    // whatever the disabled node last left behind; that staleness is the point.
    struct PassInfo
    {
        std::string name;
        bool        culled;
        bool        disabled;
    };
    [[nodiscard]] std::vector<PassInfo> GetPassInfo() const;
    void SetPassEnabled(const std::string& name, bool enabled);

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

    // The frame's submission plan: one line per run, with the queue it goes to,
    // the nodes it covers and the cross-queue wait that precedes it. This is how
    // you tell overlap from a plan that merely looks parallel.
    [[nodiscard]] std::string DumpSchedule() const;

private:
    struct ImportedResource
    {
        Resource*       tracked = nullptr; // null for raw imports
        ID3D12Resource* raw     = nullptr;
        std::string     debugName;
        bool            externallyRead = false;
    };

    // UAV hazard bookkeeping. Tracked *state* already crosses frames, because it
    // lives on the Resource and not here — but a UAV write followed by a UAV read
    // needs a barrier without any state change, and that was recorded per frame.
    // The per-frame flush covered the frame boundary; with it gone, frame N's
    // write to frame N+1's read is a real hazard (adaptive q's TileGuideQ is the
    // live one). Imports still last exactly one frame (§A2), so this is keyed by
    // the underlying resource rather than by handle.
    struct UavUsage
    {
        bool     writtenSinceLastRead = false;
        bool     readSinceLastWrite   = false;
        uint64_t lastTouchedFrame     = 0;
    };

    struct PassNode
    {
        std::string                                      name;
        std::vector<RenderGraphPassBuilder::Declaration> declarations;
        std::function<void()>                            execute;
        GraphQueue                                       queue     = GraphQueue::Direct;
        bool                                             neverCull = false;
        bool                                             culled    = false;
        bool                                             disabled  = false;
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

    // A run of consecutive compiled passes on one queue: one command list, one
    // ExecuteCommandLists. A direct run is split whenever a node inside it needs
    // to wait on the compute queue — that split is the whole point, because it is
    // what lets the direct work *before* the wait overlap the async chain.
    struct Submission
    {
        GraphQueue queue          = GraphQueue::Direct;
        uint32_t   firstCompiled  = 0;
        uint32_t   count          = 0;
        uint32_t   waitSubmission = kNoSlot; // cross-queue producer to wait for
        bool       signals        = false;   // someone later waits on this run
    };

    static D3D12_RESOURCE_STATES ToResourceState(GraphAccess access);
    static const char*           ToString(GraphAccess access);

    void Cull();

    // Groups the compiled passes into per-queue runs and works out the cross-queue
    // waits between them. Runs after barrier placement, on the compiled order.
    void Schedule();

    // Records one run into the given list, closing it. Timestamps and the query
    // resolve stay on the direct queue.
    void RecordSubmission(CommandContext& context, const Submission& submission, bool isLastSubmission);

    // A reusable list for a run the graph allocates itself (everything but the
    // first). Reset from this frame's allocator; several lists may share one
    // allocator because only one of them records at a time.
    ID3D12GraphicsCommandList4* AcquireList(GraphicsDevice& device, GraphQueue queue, uint32_t indexOnQueue);

    // Puts a synthesized transition into a slot's batch according to the placement
    // mode, and returns the slot it (or its BEGIN half) landed in.
    uint32_t PlaceTransition(const D3D12_RESOURCE_BARRIER& barrier, ID3D12Resource* resource,
                             uint32_t earliestSlot, uint32_t consumerSlot);

    [[nodiscard]] std::string DescribePlacement(uint32_t placedSlot, uint32_t consumerSlot) const;

    // Two timestamps per node; the cap is the node budget one frame may time.
    static constexpr uint32_t kMaxTimedPasses = 64;

    // "This resource has not been touched yet this frame", so the earliest legal
    // point for its first barrier is the top of the frame.
    static constexpr uint32_t kNoSlot = UINT32_MAX;

    // How long a UavUsage entry outlives its last touch. Frame pacing keeps the
    // CPU under NUM_FRAMES frames ahead, so work older than that has completed on
    // the GPU and needs no barrier — double it and dropping an entry can never
    // drop a hazard. Without this the map would accumulate dead keys across scene
    // switches and resizes.
    static constexpr uint64_t kUavUsageLifetimeFrames = Constants::Graphics::NUM_FRAMES * 2;

    std::vector<ImportedResource> m_resources;
    // Survives Reset(), unlike m_resources: see UavUsage.
    std::unordered_map<ID3D12Resource*, UavUsage> m_uavUsage;
    uint64_t                      m_frameCounter = 0;
    std::vector<PassNode>         m_passes;
    // User state, not frame state: survives Reset so a toggle sticks while the
    // graph is rebuilt every frame.
    std::set<std::string>         m_disabledPasses;
    std::vector<CompiledPass>     m_compiled;
    std::vector<Submission>       m_submissions;
    std::vector<std::string>      m_barrierLog;
    bool                          m_logBarriers = false;
    bool                          m_asyncCompute = false;
    BarrierPlacement              m_barrierPlacement = BarrierPlacement::Consumer;

    // Extra command lists the graph records runs into, kept across frames and
    // reset on acquire. Index 0 of the direct pool is never used — the frame's
    // primary list is supplied by the caller.
    std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4>> m_directLists;
    std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4>> m_computeLists;

    Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_timerQueryHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource>  m_timerReadback;
    uint64_t                                m_timerFrequency = 0;
    bool                                    m_timingEnabled  = false;
    uint32_t                                m_timedPassCount = 0;
    std::vector<std::string>                m_timedPassNames;
    std::vector<PassTiming>                 m_timings;

    // Keyed by node name rather than index: culling changes which nodes run from
    // frame to frame, so a positional accumulator would mix two different nodes.
    struct TimingAccumulator
    {
        double   totalMilliseconds = 0.0;
        float    maxMilliseconds   = 0.0f;
        uint32_t frames            = 0;
    };
    std::map<std::string, TimingAccumulator> m_timingHistory;
};
