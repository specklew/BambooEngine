#pragma once
#include <vector>

// ADR 0017 L2: owns one descriptor heap, hands out slots in it, and creates the
// views. Descriptor index arithmetic (start handle + index * increment) lives
// here and nowhere else.
struct DescriptorHandle
{
    static constexpr uint32_t InvalidIndex = ~0u;

    uint32_t                    index = InvalidIndex;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu   = {};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu   = {}; // zero on non-shader-visible heaps

    [[nodiscard]] bool IsValid() const { return index != InvalidIndex; }
};

class DescriptorAllocator
{
public:
    void Initialize(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity,
                    bool shaderVisible, const wchar_t* debugName);

    // Single slot, reusing freed ones; contiguous ranges never reuse (a range is
    // addressed as a base + offset by root-signature descriptor tables).
    DescriptorHandle Allocate();
    DescriptorHandle AllocateRange(uint32_t count);
    void Free(uint32_t index);

    [[nodiscard]] DescriptorHandle GetHandle(uint32_t index) const;
    [[nodiscard]] ID3D12DescriptorHeap* GetHeap() const { return m_heap.Get(); }
    [[nodiscard]] uint32_t GetCapacity() const { return m_capacity; }
    [[nodiscard]] uint32_t GetAllocatedCount() const;

    void CreateShaderResourceView(uint32_t index, ID3D12Resource* resource,
                                  const D3D12_SHADER_RESOURCE_VIEW_DESC& desc);
    void CreateUnorderedAccessView(uint32_t index, ID3D12Resource* resource,
                                   const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc);
    void CreateConstantBufferView(uint32_t index, const D3D12_CONSTANT_BUFFER_VIEW_DESC& desc);

    // Null views keep a slot legal to bind after its resource dies: shader reads
    // return zero and writes are discarded, instead of the descriptor pointing at
    // freed memory.
    void CreateNullShaderResourceView(uint32_t index, DXGI_FORMAT format, D3D12_SRV_DIMENSION dimension);
    void CreateNullUnorderedAccessView(uint32_t index, DXGI_FORMAT format, D3D12_UAV_DIMENSION dimension);
    void CreateNullShaderResourceView(uint32_t index, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc);
    void CreateNullConstantBufferView(uint32_t index);

private:
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_heap;
    ID3D12Device*                                m_device = nullptr;

    D3D12_CPU_DESCRIPTOR_HANDLE m_cpuStart = {};
    D3D12_GPU_DESCRIPTOR_HANDLE m_gpuStart = {};

    uint32_t              m_capacity      = 0;
    uint32_t              m_increment     = 0;
    uint32_t              m_nextFreeIndex = 0;
    std::vector<uint32_t> m_freeIndices;
    bool                  m_shaderVisible = false;
};
