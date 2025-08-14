#include "pch.h"
#include "DescriptorHeapAllocator.h"

/// Descriptor heap allocator manages a descriptor heap by keeping track of free indices.
/// It does not manage the lifetime of the descriptor heap itself.
/// The descriptor heap must be created and released by the user.
/// 
/// @param device The D3D12 device used to create the descriptor heap.
/// @param descriptorHeap Descriptor heap to manage.
DescriptorHeapAllocator::DescriptorHeapAllocator(ID3D12Device* device, ID3D12DescriptorHeap* descriptorHeap) :
    m_device(device),
    m_descriptorHeap(descriptorHeap)
{
    assert(m_device != nullptr && "Device cannot be null");
    assert(m_descriptorHeap != nullptr && "Descriptor heap cannot be null");

    m_descriptorStartCpuHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    m_descriptorStartGpuHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    
    const D3D12_DESCRIPTOR_HEAP_DESC desc = descriptorHeap->GetDesc();
    
    m_handleIncrementSize = m_device->GetDescriptorHandleIncrementSize(desc.Type);
    m_freeIndices.reserve(desc.NumDescriptors);
    
    for (int i = 0; i < desc.NumDescriptors; i++)
    {
        m_freeIndices.push_back(i);
    }
}

DescriptorHeapAllocator::~DescriptorHeapAllocator()
{
    // Useless.. TODO: consider allocating descriptor heaps in the constructor and releasing them in the destructor.
    m_descriptorHeap = nullptr;
    m_freeIndices.clear();
}

void DescriptorHeapAllocator::Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpuHandle,
    D3D12_GPU_DESCRIPTOR_HANDLE* outGpuHandle)
{
    assert(m_freeIndices.size() > 0 && "No free indices available in the descriptor heap allocator");

    const int index = m_freeIndices.back();
    m_freeIndices.pop_back();

    if (outCpuHandle != nullptr) outCpuHandle->ptr = m_descriptorStartCpuHandle.ptr + static_cast<long long>(index * m_handleIncrementSize);
    if (outGpuHandle != nullptr) outGpuHandle->ptr = m_descriptorStartGpuHandle.ptr + static_cast<long long>(index * m_handleIncrementSize);
}

void DescriptorHeapAllocator::Free(const D3D12_CPU_DESCRIPTOR_HANDLE& cpuHandle, const D3D12_GPU_DESCRIPTOR_HANDLE& gpuHandle)
{
    UINT64 gpuIndex = (gpuHandle.ptr - m_descriptorStartGpuHandle.ptr) / m_handleIncrementSize;
    UINT64 cpuIndex = (cpuHandle.ptr - m_descriptorStartCpuHandle.ptr) / m_handleIncrementSize;
    assert(gpuIndex == cpuIndex && "CPU and GPU handles do not match");

    m_freeIndices.push_back(gpuIndex);
}




