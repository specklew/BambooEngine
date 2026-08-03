#pragma once
#include "GlobalDescriptorHeap.h"
#include "GraphAccess.h"

// One shader binding, declared once and used four times: to build the root
// signature, to bind at draw time, to answer "is this register mine?" (ADR 0019),
// and to tell the render graph what the pass does to whatever is bound there
// (ADR 0017 step 3). Slots are `static constexpr` next to the pass that owns them.
//
// A slot says *where* — kind, register, space, and for table entries which heap
// offset — and *how it is used*. It still does not say *what*: the resource is
// supplied per frame, because the graph's imports last exactly one frame.
enum class BindingKind : uint8_t
{
    Cbv,
    Srv,
    Uav,
    Sampler
};

enum class BindingStorage : uint8_t
{
    RootDescriptor, // a raw GPU virtual address, 2 DWORDs of root signature
    RootConstants,  // values inline in the command list, 1 DWORD each
    Table,          // an offset into whichever descriptor heap is bound
    StaticSampler   // baked into the signature; no root parameter, no bind call
};

inline constexpr uint32_t kUnboundedRegisterCount = ~0u;

// space0 is the frame layout; everything a pass declares lives in space1
// (ADR 0017 phase 4). The split is what makes frame-vs-pass structural rather
// than conventional: a pass register can never collide with a frame one, and no
// longer has to be chosen around the frame numbers. The Pass* constructors below
// mirror the BAMBOO_PASS_* macros the shaders use, so a declaration reads the
// same on both sides.
inline constexpr uint32_t kFrameRegisterSpace = 0;
inline constexpr uint32_t kPassRegisterSpace  = 1;

struct BindingSlot
{
    const char*    name           = "";
    BindingKind    kind           = BindingKind::Srv;
    uint32_t       shaderRegister = 0;
    uint32_t       registerSpace  = 0;
    uint32_t       registerCount  = 1;
    BindingStorage storage        = BindingStorage::RootDescriptor;
    uint32_t       constantCount  = 0; // RootConstants only
    uint32_t       tableIndex     = 0; // Table only: which table parameter
    uint32_t       heapOffset     = 0; // Table only: descriptors from table start

    // What the graph records for this binding. None means the graph never sees it.
    GraphAccess    graphAccess    = GraphAccess::None;
    bool           graphWrites    = false;
};

// A slot's graph access, declared next to its register rather than repeated at
// the node. RenderGraphPassBuilder::Declare reads both fields, so the direction
// lives here too: GraphAccess alone cannot say whether a RenderTarget is being
// written or merely kept.
constexpr BindingSlot GraphReads(BindingSlot slot, GraphAccess access)
{
    slot.graphAccess = access;
    slot.graphWrites = false;
    return slot;
}

constexpr BindingSlot GraphWrites(BindingSlot slot, GraphAccess access)
{
    slot.graphAccess = access;
    slot.graphWrites = true;
    return slot;
}

// C++17 has no designated initializers, so slots are built through these rather
// than positionally — a nine-field aggregate literal is unreadable and easy to
// mis-order.
constexpr BindingSlot RootCbv(const char* name, uint32_t shaderRegister, uint32_t registerSpace = 0)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = BindingKind::Cbv;
    slot.shaderRegister = shaderRegister;
    slot.registerSpace  = registerSpace;
    return slot;
}

constexpr BindingSlot RootSrv(const char* name, uint32_t shaderRegister, uint32_t registerSpace = 0)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = BindingKind::Srv;
    slot.shaderRegister = shaderRegister;
    slot.registerSpace  = registerSpace;
    return slot;
}

constexpr BindingSlot RootUav(const char* name, uint32_t shaderRegister, uint32_t registerSpace = 0)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = BindingKind::Uav;
    slot.shaderRegister = shaderRegister;
    slot.registerSpace  = registerSpace;
    return slot;
}

// A static sampler: part of the signature, but not a root parameter, so it costs
// no DWORDs and is never bound at draw time.
constexpr BindingSlot Sampler(const char* name, uint32_t shaderRegister)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = BindingKind::Sampler;
    slot.shaderRegister = shaderRegister;
    slot.storage        = BindingStorage::StaticSampler;
    return slot;
}

constexpr BindingSlot RootConstants(const char* name, uint32_t shaderRegister, uint32_t constantCount,
                                    uint32_t registerSpace = 0)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = BindingKind::Cbv;
    slot.shaderRegister = shaderRegister;
    slot.registerSpace  = registerSpace;
    slot.storage        = BindingStorage::RootConstants;
    slot.constantCount  = constantCount;
    return slot;
}

// Table entry addressed by global-heap slot.
constexpr BindingSlot TableEntry(const char* name, BindingKind kind, uint32_t shaderRegister,
                                 GlobalDescriptor heapSlot, uint32_t registerCount = 1, uint32_t tableIndex = 0)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = kind;
    slot.shaderRegister = shaderRegister;
    slot.registerCount  = registerCount;
    slot.storage        = BindingStorage::Table;
    slot.tableIndex     = tableIndex;
    slot.heapOffset     = GlobalDescriptorHeap::IndexOf(heapSlot);
    return slot;
}

// Table entry addressed by raw offset — a pass that owns a private descriptor
// heap, where offsets are heap-start-relative and mean nothing globally.
constexpr BindingSlot TableEntryAt(const char* name, BindingKind kind, uint32_t shaderRegister, uint32_t heapOffset,
                                   uint32_t registerCount = 1, uint32_t tableIndex = 0)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = kind;
    slot.shaderRegister = shaderRegister;
    slot.registerCount  = registerCount;
    slot.storage        = BindingStorage::Table;
    slot.tableIndex     = tableIndex;
    slot.heapOffset     = heapOffset;
    return slot;
}

// ---------------------------------------------------------------------------
// Pass-scoped slots — the space1 counterparts of everything above. A pass
// declares only these; the frame layout is the sole occupant of space0.
// ---------------------------------------------------------------------------

constexpr BindingSlot InPassSpace(BindingSlot slot)
{
    slot.registerSpace = kPassRegisterSpace;
    return slot;
}

constexpr BindingSlot PassCbv(const char* name, uint32_t shaderRegister)
{
    return RootCbv(name, shaderRegister, kPassRegisterSpace);
}

constexpr BindingSlot PassSrv(const char* name, uint32_t shaderRegister)
{
    return RootSrv(name, shaderRegister, kPassRegisterSpace);
}

constexpr BindingSlot PassUav(const char* name, uint32_t shaderRegister)
{
    return RootUav(name, shaderRegister, kPassRegisterSpace);
}

constexpr BindingSlot PassRootConstants(const char* name, uint32_t shaderRegister, uint32_t constantCount)
{
    return RootConstants(name, shaderRegister, constantCount, kPassRegisterSpace);
}

constexpr BindingSlot PassTableEntry(const char* name, BindingKind kind, uint32_t shaderRegister,
                                     GlobalDescriptor heapSlot, uint32_t registerCount = 1, uint32_t tableIndex = 0)
{
    return InPassSpace(TableEntry(name, kind, shaderRegister, heapSlot, registerCount, tableIndex));
}

constexpr BindingSlot PassTableEntryAt(const char* name, BindingKind kind, uint32_t shaderRegister, uint32_t heapOffset,
                                       uint32_t registerCount = 1, uint32_t tableIndex = 0)
{
    return InPassSpace(TableEntryAt(name, kind, shaderRegister, heapOffset, registerCount, tableIndex));
}
