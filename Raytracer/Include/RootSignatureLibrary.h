#pragma once
#include <string>
#include <unordered_map>
#include <vector>

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
    Microsoft::WRL::ComPtr<ID3D12RootSignature> Create(ID3D12Device*                    device,
                                                       const D3D12_ROOT_SIGNATURE_DESC& desc,
                                                       const wchar_t*                   debugName,
                                                       bool usesFrameLayout = false);

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
    struct Entry
    {
        // Holds a reference so the map key can never be a recycled address.
        Microsoft::WRL::ComPtr<ID3D12RootSignature> signature;
        std::vector<RootSignatureBinding>           bindings;
        std::vector<bool>                           bindingReferenced;
        std::string                                 debugName;
        bool                                        usesFrameLayout = false;
    };

    std::unordered_map<ID3D12RootSignature*, Entry> m_signatures;
};
