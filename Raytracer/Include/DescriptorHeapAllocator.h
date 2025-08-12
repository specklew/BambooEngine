#pragma once

/// DESCRIPTOR HEAP ALLOCATOR
/// Currently used only for ImGui implementation.
/// Other descriptor heaps are being preallocated, but this behaviour may change in the future if there is a need.

class DescriptorHeapAllocator
{
public:
    explicit DescriptorHeapAllocator(ID3D12Device* device, ID3D12DescriptorHeap* descriptorHeap);
    ~DescriptorHeapAllocator();
    
    DescriptorHeapAllocator(const DescriptorHeapAllocator&) = delete;
    DescriptorHeapAllocator(DescriptorHeapAllocator&&) = delete;
    DescriptorHeapAllocator& operator=(const DescriptorHeapAllocator&) = delete;
    DescriptorHeapAllocator& operator=(DescriptorHeapAllocator&&) = delete;

    void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE* outGpuHandle);
    void Free(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle);
private:
    ID3D12Device* m_device;
    ID3D12DescriptorHeap* m_descriptorHeap;
    
    D3D12_CPU_DESCRIPTOR_HANDLE m_descriptorStartCpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE m_descriptorStartGpuHandle;

    UINT m_handleIncrementSize;

    std::vector<UINT> m_freeIndices;
};
