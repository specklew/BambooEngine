#pragma once
#include <functional>
#include <string>
#include <vector>

class CommandContext;
class Resource;

// ADR 0017 L5, step A: passes declare what they read and write; the graph turns
// those declarations into barriers and runs the passes in dependency order.
// Bindings are untouched at this step — a converted pass records exactly what it
// recorded before, minus its hand-placed barriers.

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
    Present
};

class RenderGraphPassBuilder
{
public:
    void Read(GraphResourceHandle resource, GraphAccess access);
    void Write(GraphResourceHandle resource, GraphAccess access);

private:
    friend class RenderGraph;

    struct Declaration
    {
        GraphResourceHandle resource;
        GraphAccess         access;
        bool                isWrite;
    };

    std::vector<Declaration> m_declarations;
};

class RenderGraph
{
public:
    // Resources are imported, not owned: transient allocation and aliasing are
    // phase 6. A tracked Resource keeps its phase-0 state model; raw ComPtr
    // resources (the VXPG textures) carry no state and only get UAV barriers.
    GraphResourceHandle Import(Resource& resource, const char* debugName);
    GraphResourceHandle ImportRaw(ID3D12Resource* resource, const char* debugName);

    void AddPass(const char* name,
                 const std::function<void(RenderGraphPassBuilder&)>& declare,
                 std::function<void()> execute);

    // Emits the synthesized barriers and runs each surviving pass in order.
    void Execute(CommandContext& context);

    // Clears passes and declarations; imported resources stay registered so the
    // per-frame rebuild does not re-import the same textures every frame.
    void Reset();

    // Barrier attribution for a perf delta (ADR 0017: this exists to explain a
    // regression, not to gate on byte-identical output).
    [[nodiscard]] std::string DumpBarriers() const;

private:
    struct ImportedResource
    {
        Resource*             tracked = nullptr; // null for raw imports
        ID3D12Resource*       raw     = nullptr;
        std::string           debugName;
        D3D12_RESOURCE_STATES stateInGraph = D3D12_RESOURCE_STATE_COMMON;
        bool                  hasStateInGraph = false;
        bool                  writtenSinceLastRead = false;
    };

    struct PassNode
    {
        std::string                                m_name;
        std::vector<RenderGraphPassBuilder::Declaration> declarations;
        std::function<void()>                      execute;
    };

    static D3D12_RESOURCE_STATES ToResourceState(GraphAccess access);

    std::vector<ImportedResource> m_resources;
    std::vector<PassNode>         m_passes;
    std::vector<std::string>      m_barrierLog;
};
