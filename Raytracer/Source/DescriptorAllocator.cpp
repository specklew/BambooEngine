#include "pch.h"
#include "DescriptorAllocator.h"

#include "Utils/Utils.h"

void DescriptorAllocator::Initialize(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity,
                                     bool shaderVisible, const wchar_t* debugName)
{
    assert(device && "Device cannot be null when creating a descriptor allocator");
    assert(capacity > 0 && "Descriptor heap capacity must be non-zero");

    m_device        = device;
    m_capacity      = capacity;
    m_shaderVisible = shaderVisible;

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type           = type;
    desc.NumDescriptors = capacity;
    desc.Flags          = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                                        : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    desc.NodeMask       = 0;

    ThrowIfFailed(m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_heap)));
    if (debugName)
        m_heap->SetName(debugName);

    m_increment = m_device->GetDescriptorHandleIncrementSize(type);
    m_cpuStart  = m_heap->GetCPUDescriptorHandleForHeapStart();
    if (shaderVisible)
        m_gpuStart = m_heap->GetGPUDescriptorHandleForHeapStart();

    m_nextFreeIndex = 0;
    m_freeIndices.clear();
}

DescriptorHandle DescriptorAllocator::Allocate()
{
    if (!m_freeIndices.empty())
    {
        const uint32_t index = m_freeIndices.back();
        m_freeIndices.pop_back();
        return GetHandle(index);
    }

    return AllocateRange(1);
}

DescriptorHandle DescriptorAllocator::AllocateRange(uint32_t count)
{
    assert(count > 0 && "Descriptor range must hold at least one descriptor");

    if (m_nextFreeIndex + count > m_capacity)
    {
        spdlog::error("Descriptor heap out of space: {} descriptors requested, {} of {} used", count, m_nextFreeIndex, m_capacity);
        assert(false && "Descriptor heap out of space");
        return {};
    }

    const uint32_t base = m_nextFreeIndex;
    m_nextFreeIndex += count;
    return GetHandle(base);
}

void DescriptorAllocator::Free(uint32_t index)
{
    assert(index < m_nextFreeIndex && "Freeing a descriptor slot that was never allocated");
    m_freeIndices.push_back(index);
}

DescriptorHandle DescriptorAllocator::GetHandle(uint32_t index) const
{
    assert(index < m_capacity && "Descriptor index out of heap range");

    DescriptorHandle handle;
    handle.index   = index;
    handle.cpu.ptr = m_cpuStart.ptr + static_cast<SIZE_T>(index) * m_increment;
    if (m_shaderVisible)
        handle.gpu.ptr = m_gpuStart.ptr + static_cast<UINT64>(index) * m_increment;

    return handle;
}

uint32_t DescriptorAllocator::GetAllocatedCount() const
{
    return m_nextFreeIndex - static_cast<uint32_t>(m_freeIndices.size());
}

void DescriptorAllocator::CreateShaderResourceView(uint32_t index, ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc)
{
    m_device->CreateShaderResourceView(resource, &desc, GetHandle(index).cpu);
}

void DescriptorAllocator::CreateUnorderedAccessView(uint32_t index, ID3D12Resource* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc)
{
    m_device->CreateUnorderedAccessView(resource, nullptr, &desc, GetHandle(index).cpu);
}

void DescriptorAllocator::CreateConstantBufferView(uint32_t index, const D3D12_CONSTANT_BUFFER_VIEW_DESC& desc)
{
    m_device->CreateConstantBufferView(&desc, GetHandle(index).cpu);
}

void DescriptorAllocator::CreateNullShaderResourceView(uint32_t index, DXGI_FORMAT format, D3D12_SRV_DIMENSION dimension)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
    desc.Format                  = format;
    desc.ViewDimension           = dimension;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (dimension == D3D12_SRV_DIMENSION_TEXTURE2D)
        desc.Texture2D.MipLevels = 1;

    m_device->CreateShaderResourceView(nullptr, &desc, GetHandle(index).cpu);
}

void DescriptorAllocator::CreateNullShaderResourceView(uint32_t index, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc)
{
    m_device->CreateShaderResourceView(nullptr, &desc, GetHandle(index).cpu);
}

void DescriptorAllocator::CreateNullConstantBufferView(uint32_t index)
{
    // BufferLocation 0 is the documented null CBV: reads return zero.
    const D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
    m_device->CreateConstantBufferView(&desc, GetHandle(index).cpu);
}

void DescriptorAllocator::CreateNullUnorderedAccessView(uint32_t index, DXGI_FORMAT format, D3D12_UAV_DIMENSION dimension)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
    desc.Format        = format;
    desc.ViewDimension = dimension;
    if (dimension == D3D12_UAV_DIMENSION_TEXTURE3D)
        desc.Texture3D.WSize = 1;

    m_device->CreateUnorderedAccessView(nullptr, nullptr, &desc, GetHandle(index).cpu);
}
