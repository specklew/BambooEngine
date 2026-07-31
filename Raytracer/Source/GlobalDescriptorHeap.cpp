#include "pch.h"
#include "GlobalDescriptorHeap.h"

namespace
{
    // The view a slot gets when it has no live resource. The format must match how
    // the shader declares the binding (float/uint/int) — a null view returns zero,
    // but the type still has to line up.
    enum class NullViewKind
    {
        None, // filled during initialization before anything is dispatched
        SrvTexture2D,
        UavTexture2D,
        UavTexture3D
    };

    struct NullView
    {
        NullViewKind kind;
        DXGI_FORMAT  format;
    };

    constexpr NullView NullViews[] = {
        { NullViewKind::None,         DXGI_FORMAT_UNKNOWN },                 // ImGuiFont
        { NullViewKind::None,         DXGI_FORMAT_UNKNOWN },                 // CameraMatrices
        { NullViewKind::None,         DXGI_FORMAT_UNKNOWN },                 // RaytraceOutput
        { NullViewKind::None,         DXGI_FORMAT_UNKNOWN },                 // Tlas
        { NullViewKind::None,         DXGI_FORMAT_UNKNOWN },                 // Vertices
        { NullViewKind::None,         DXGI_FORMAT_UNKNOWN },                 // Indices
        { NullViewKind::SrvTexture2D, DXGI_FORMAT_R8G8B8A8_UNORM },          // MaterialTextures
        { NullViewKind::SrvTexture2D, DXGI_FORMAT_R8G8B8A8_UNORM },          // Skybox
        { NullViewKind::UavTexture3D, DXGI_FORMAT_R32_UINT },                // VoxelOccupancy
        { NullViewKind::UavTexture3D, DXGI_FORMAT_R32_UINT },                // VoxelIrradiance
        { NullViewKind::UavTexture3D, DXGI_FORMAT_R32_UINT },                // VoxelVplCount
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32B32A32_FLOAT },      // ShadingPoints
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32_SINT },                // SuperpixelIndex
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32B32A32_FLOAT },      // SuperpixelCenter
        { NullViewKind::UavTexture3D, DXGI_FORMAT_R32G32B32A32_FLOAT },      // VoxelRepresentative
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32B32A32_FLOAT },      // VplPosition
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32B32A32_UINT },       // VBuffer
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32_SINT },             // SpixelGathered
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32_UINT },                // SpixelCounter
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32_UINT },                // ClusterVisibilityMask
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32B32A32_FLOAT },      // FuzzyWeight
        { NullViewKind::UavTexture2D, DXGI_FORMAT_R32G32B32A32_SINT },       // FuzzyIndex
    };

    static_assert(std::size(NullViews) == static_cast<size_t>(GlobalDescriptor::Count),
                  "Every GlobalDescriptor needs a null view definition");
}

GlobalDescriptorHeap& GlobalDescriptorHeap::Get()
{
    static GlobalDescriptorHeap instance;
    return instance;
}

void GlobalDescriptorHeap::Initialize(ID3D12Device* device)
{
    assert(!m_initialized && "Global descriptor heap initialized twice");

    m_allocator.Initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Capacity(), true,
                           L"Global Descriptor Heap");

    for (uint32_t slot = 0; slot < static_cast<uint32_t>(GlobalDescriptor::Count); ++slot)
    {
        const DescriptorHandle handle = m_allocator.AllocateRange(GlobalDescriptorSlotCounts[slot]);
        assert(handle.index == IndexOf(static_cast<GlobalDescriptor>(slot)) &&
               "Heap layout diverged from the compile-time indices the root signatures use");
    }

    // Passes fill their slots as they initialize and can drop them again on resize
    // or scene switch; starting every one of them null means the heap never holds a
    // descriptor pointing at nothing.
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(GlobalDescriptor::Count); ++slot)
        ClearSlot(static_cast<GlobalDescriptor>(slot));

    m_allocatedMaterialTextureSlots = 0;
    m_initialized                   = true;
}

D3D12_CPU_DESCRIPTOR_HANDLE GlobalDescriptorHeap::CpuHandle(GlobalDescriptor slot, uint32_t offset) const
{
    assert(offset < GlobalDescriptorSlotCounts[static_cast<uint32_t>(slot)] &&
           "Descriptor offset past the end of its slot range");
    return m_allocator.GetHandle(IndexOf(slot) + offset).cpu;
}

D3D12_GPU_DESCRIPTOR_HANDLE GlobalDescriptorHeap::GpuStart() const
{
    return m_allocator.GetHandle(0).gpu;
}

int GlobalDescriptorHeap::AllocateMaterialTextureSlot()
{
    if (m_allocatedMaterialTextureSlots >= Constants::Graphics::MAX_TEXTURES)
    {
        spdlog::error("Material texture slots exhausted ({} available)", Constants::Graphics::MAX_TEXTURES);
        return -1;
    }

    return static_cast<int>(m_allocatedMaterialTextureSlots++);
}

void GlobalDescriptorHeap::ReleaseMaterialTextureSlots()
{
    const NullView& nullView = NullViews[static_cast<uint32_t>(GlobalDescriptor::MaterialTextures)];

    for (uint32_t slot = 0; slot < m_allocatedMaterialTextureSlots; ++slot)
        m_allocator.CreateNullShaderResourceView(IndexOf(GlobalDescriptor::MaterialTextures) + slot,
                                                 nullView.format, D3D12_SRV_DIMENSION_TEXTURE2D);

    m_allocatedMaterialTextureSlots = 0;
}

void GlobalDescriptorHeap::ClearSlot(GlobalDescriptor slot)
{
    const NullView& nullView = NullViews[static_cast<uint32_t>(slot)];
    const uint32_t  base     = IndexOf(slot);
    const uint32_t  count    = GlobalDescriptorSlotCounts[static_cast<uint32_t>(slot)];

    for (uint32_t i = 0; i < count; ++i)
    {
        switch (nullView.kind)
        {
        case NullViewKind::SrvTexture2D:
            m_allocator.CreateNullShaderResourceView(base + i, nullView.format, D3D12_SRV_DIMENSION_TEXTURE2D);
            break;
        case NullViewKind::UavTexture2D:
            m_allocator.CreateNullUnorderedAccessView(base + i, nullView.format, D3D12_UAV_DIMENSION_TEXTURE2D);
            break;
        case NullViewKind::UavTexture3D:
            m_allocator.CreateNullUnorderedAccessView(base + i, nullView.format, D3D12_UAV_DIMENSION_TEXTURE3D);
            break;
        case NullViewKind::None:
            break;
        }
    }
}
