#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "BindingSlot.h"

// A single (type, register range) the signature makes available to a shader.
// A root descriptor and a one-entry table look identical here on purpose: what
// matters for validation is which registers a shader may legally reference.
struct RootSignatureBinding
{
    D3D12_DESCRIPTOR_RANGE_TYPE type          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    uint32_t                    baseRegister  = 0;
    uint32_t                    registerSpace = 0;
    uint32_t                    registerCount = 1;
    uint32_t                    rootParameterIndex = 0;
    bool                        isDescriptorTable  = false;
};

// The slot table a signature was built from, plus where each slot landed. Owned
// by RootSignatureLibrary and pointer-stable, so a RootSignature can hold it.
class RootSignatureLayout
{
public:
    // Root parameter index for a slot, matched on (kind, register, space).
    // Asserts if the slot was never added — binding through the wrong signature.
    [[nodiscard]] uint32_t RootParameterOf(const BindingSlot& slot) const;

    [[nodiscard]] bool IsGraphics() const { return m_graphics; }

private:
    friend class RootSignatureBuilder;

    // Root-descriptor and root-constant slots resolve through a dense table
    // rather than a scan: Set() runs a few dozen times per pass per frame, and a
    // Debug-build scan over the slot list was measurable. Table slots are not in
    // here — they bind by table index, not by register.
    static constexpr uint32_t kLookupSpaces    = 2;
    static constexpr uint32_t kLookupKinds     = 4;
    static constexpr uint32_t kLookupRegisters = 32;
    static constexpr uint8_t  kNoRootParameter = 0xFF;

    std::vector<BindingSlot> m_slots;
    std::vector<uint32_t>    m_rootParameterIndices; // parallel to m_slots
    std::string              m_debugName;
    bool                     m_graphics = false;
    uint8_t                  m_lookup[kLookupSpaces][kLookupKinds][kLookupRegisters] = {};
};

// A built root signature that remembers its slot table, so passes bind by slot
// instead of by a root parameter index they had to track themselves. The bind
// verb follows from the slot's kind, so a CBV cannot be bound as an SRV.
class RootSignature
{
public:
    RootSignature() = default;

    [[nodiscard]] ID3D12RootSignature* Get() const { return m_signature.Get(); }
    explicit operator bool() const { return m_signature != nullptr; }

    void SetTable(ID3D12GraphicsCommandList* commandList, uint32_t tableIndex, D3D12_GPU_DESCRIPTOR_HANDLE heapStart) const;
    void Set(ID3D12GraphicsCommandList* commandList, const BindingSlot& slot, D3D12_GPU_VIRTUAL_ADDRESS address) const;
    void SetConstants(ID3D12GraphicsCommandList* commandList, const BindingSlot& slot, const void* values, uint32_t numValues) const;

private:
    friend class RootSignatureBuilder;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_signature;
    const RootSignatureLayout*                  m_layout = nullptr;
};

// Builds a root signature from slots. Parameter order is fixed and knowable
// while adding: the tableCount table parameters come first, then root
// parameters in declaration order — so Add() can resolve an index immediately
// and nothing has to be patched up at the end.
class RootSignatureBuilder
{
public:
    RootSignatureBuilder(const wchar_t* debugName, uint32_t tableCount);

    // The space0 frame layout: its table entries and its root descriptors, in
    // the fixed order every raytracing signature shares.
    RootSignatureBuilder& AddFrameLayout();

    RootSignatureBuilder& Add(const BindingSlot& slot);

    // A pass's whole slot table in one call. The array form takes the table by
    // reference so the count comes from the type and cannot drift.
    RootSignatureBuilder& Add(const BindingSlot* slots, size_t count);
    template <size_t N>
    RootSignatureBuilder& Add(const BindingSlot (&slots)[N])
    {
        return Add(slots, N);
    }

    RootSignatureBuilder& WithStaticSamplers();
    RootSignatureBuilder& WithFlags(D3D12_ROOT_SIGNATURE_FLAGS flags);
    RootSignatureBuilder& ForGraphics(); // binds through SetGraphicsRoot* instead

    RootSignature Build(ID3D12Device* device);

private:
    std::vector<BindingSlot>            m_slots;
    std::vector<uint32_t>               m_rootParameterIndices;
    std::wstring                        m_debugName;
    uint32_t                            m_tableCount        = 0;
    uint32_t                            m_nextRootParameter = 0;
    bool                                m_usesFrameLayout   = false;
    bool                                m_graphics          = false;
    bool                                m_staticSamplers    = false;
    D3D12_ROOT_SIGNATURE_FLAGS          m_flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;
};

// ADR 0017 L4: every root signature in the engine is created here, because D3D12
// offers no way to read a layout back off an ID3D12RootSignature and phase 4
// needs it to check shader reflection against what the signature really provides.
// Creation is where the layout is still known, so it is recorded there.
class RootSignatureLibrary
{
public:
    static constexpr uint32_t kUnboundedRegisterCount = ~0u;

    static RootSignatureLibrary& Get();

    // Serializes, creates, names and records the layout. Throws on failure, after
    // logging the serializer's error blob — a malformed root signature is a
    // programming error with a message worth reading.
    //
    // usesFrameLayout marks a signature built on FrameBindingLayout: its frame
    // registers are present whether the pass reads them or not, so the
    // unused-binding report skips them there (and only there).
    Microsoft::WRL::ComPtr<ID3D12RootSignature> Create(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& desc,
                                                       const wchar_t* debugName, bool usesFrameLayout = false);

    // Null for a signature created outside the library.
    [[nodiscard]] const std::vector<RootSignatureBinding>* FindLayout(ID3D12RootSignature* signature) const;
    [[nodiscard]] std::string                              FindName(ID3D12RootSignature* signature) const;

    // Reflection validation reports which bindings a shader referenced. One
    // signature usually serves several kernels, so a binding counts as used once
    // any of them touches it — asking per shader would flag every kernel's share
    // of a shared signature as waste.
    void MarkBindingReferenced(ID3D12RootSignature* signature, size_t bindingIndex);

    // Bindings no shader ever referenced: wasted root slots, and the list phase 4
    // collapses. Call once everything has been built.
    void LogUnreferencedBindings() const;

private:
    friend class RootSignatureBuilder;

    // Attaches the slot table a builder produced. Returns a pointer that stays
    // valid for the process: unordered_map only invalidates on erase, and
    // entries are never erased.
    const RootSignatureLayout* AttachLayout(ID3D12RootSignature* signature, RootSignatureLayout&& layout);

    struct Entry
    {
        // Holds a reference so the map key can never be a recycled address.
        Microsoft::WRL::ComPtr<ID3D12RootSignature> signature;
        std::vector<RootSignatureBinding>           bindings;
        std::vector<bool>                           bindingReferenced;
        std::string                                 debugName;
        bool                                        usesFrameLayout = false;
        RootSignatureLayout                         slotLayout;
    };

    std::unordered_map<ID3D12RootSignature*, Entry> m_signatures;
};
