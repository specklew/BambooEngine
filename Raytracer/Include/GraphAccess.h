#pragma once
#include <cstdint>

// The render graph's vocabulary, split out from RenderGraph.h so a binding slot
// table can name an access without pulling the graph in (ADR 0017 step 3).

using GraphResourceHandle = uint32_t;
inline constexpr GraphResourceHandle InvalidGraphResource = ~0u;

// What a pass does to a resource. The graph maps these to the legacy states the
// barrier needs; a Write followed by a Read is the edge that orders two passes.
enum class GraphAccess
{
    None,                 // not a graph-tracked binding: constants, samplers, and
                          // the frame layout's scene buffers, which have no
                          // producer inside the frame
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
