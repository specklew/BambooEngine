#pragma once
#include "GlobalDescriptorHeap.h"

// One shader binding, declared once and used three times: to build the root
// signature, to bind at draw time, and to answer "is this register mine?"
// (ADR 0019). Slots are `static constexpr` next to the pass that owns them.
//
// A slot says *where* — kind, register, space, and for table entries which heap
// offset. It does not say *what*: the resource is supplied at bind time, until
// the render graph can resolve it (ADR 0017 step B).
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
};

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

constexpr BindingSlot RootConstants(const char* name, uint32_t shaderRegister, uint32_t constantCount)
{
    BindingSlot slot;
    slot.name           = name;
    slot.kind           = BindingKind::Cbv;
    slot.shaderRegister = shaderRegister;
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
