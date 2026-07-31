#pragma once

// Tracked legacy resource state (ADR 0017, phase 0). The legacy
// D3D12_RESOURCE_STATES value is the stored truth; ResourceStateConversion in
// ResourceStateTracker.h derives the enhanced-barrier access/layout form from it,
// so phase-3 barrier synthesis can switch backends without touching this model.
// The D3D12 implicit promotion/decay rules live here, next to the state they act on.
struct ResourceState
{
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    bool tracked            = false; // false: upload/readback heap — fixed state, never barriered
    bool isBuffer           = false;
    bool simultaneousAccess = false;
    bool promotedReadOnly   = false; // state reached via implicit promotion to a read-only state

    static constexpr D3D12_RESOURCE_STATES ReadOnlyStatesMask =
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER |
        D3D12_RESOURCE_STATE_INDEX_BUFFER |
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT |
        D3D12_RESOURCE_STATE_COPY_SOURCE |
        D3D12_RESOURCE_STATE_DEPTH_READ |
        D3D12_RESOURCE_STATE_RESOLVE_SOURCE |
        D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;

    static constexpr bool IsReadOnlyState(D3D12_RESOURCE_STATES s)
    {
        return s != D3D12_RESOURCE_STATE_COMMON && (s & ~ReadOnlyStatesMask) == 0;
    }

    // Implicit promotion out of COMMON on first GPU access: buffers and
    // simultaneous-access textures promote to any state; other textures only to
    // the copy/SRV set.
    bool CanPromoteFromCommon(D3D12_RESOURCE_STATES target) const
    {
        if (target == D3D12_RESOURCE_STATE_COMMON)
        {
            return false;
        }
        if (isBuffer || simultaneousAccess)
        {
            return true;
        }
        constexpr D3D12_RESOURCE_STATES promotableTextureStates =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_COPY_DEST |
            D3D12_RESOURCE_STATE_COPY_SOURCE;
        return (target & ~promotableTextureStates) == 0;
    }

    // Implicit decay to COMMON at every ExecuteCommandLists completion.
    bool DecaysAtExecuteCompletion() const
    {
        return isBuffer || simultaneousAccess || promotedReadOnly;
    }
};

class Resource
{
public:
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Device5> GetDevice() const { return m_device; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Resource> GetUnderlyingResource() const { return m_resource; }

    [[nodiscard]] D3D12_RESOURCE_DESC GetDesc() const
    {
        if (m_resource)
        {
            return m_resource->GetDesc();
        }
        return D3D12_RESOURCE_DESC{};
    }

    [[nodiscard]] bool CheckFormatSupport(D3D12_FORMAT_SUPPORT1 formatSupport1) const;
    [[nodiscard]] bool CheckFormatSupport(D3D12_FORMAT_SUPPORT2 formatSupport2) const;

    void SetResourceName(const std::wstring& name)
    {
        m_resourceName = name;
        if (m_resource)
        {
            m_resource->SetName(name.c_str());
        }
    }

    void SetResourceName(const std::string& name)
    {
        const auto w_name = std::wstring(name.begin(), name.end());
        SetResourceName(w_name);
    }

    [[nodiscard]] std::wstring GetResourceName() const { return m_resourceName; }

    [[nodiscard]] ResourceState& GetTrackedState() { return m_trackedState; }
    [[nodiscard]] const ResourceState& GetTrackedState() const { return m_trackedState; }

    // ADR 0017 phase-0 assert harness: emits exactly the barrier the call site always
    // emitted (expectedBefore → after) and checks expectedBefore against the tracked
    // state. The expectedBefore parameter dies at phase 3.
    void TransitionChecked(ID3D12GraphicsCommandList* commandList,
                           D3D12_RESOURCE_STATES expectedBefore,
                           D3D12_RESOURCE_STATES after);
    void UavBarrierChecked(ID3D12GraphicsCommandList* commandList);

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

protected:

    Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc,
             const D3D12_CLEAR_VALUE* clearValue = nullptr,
             D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON);
    Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource> &resource,
             const D3D12_CLEAR_VALUE* clearValue = nullptr,
             D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON);

    virtual ~Resource();

    Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT m_formatSupport;
    std::unique_ptr<D3D12_CLEAR_VALUE> m_clearValue;
    std::wstring m_resourceName;
    ResourceState m_trackedState;

    void QueryFeatureSupport();

private:
    void InitializeTrackedState(D3D12_RESOURCE_STATES initialState);
};
