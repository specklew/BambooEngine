#pragma once
#include "Constants.h"
#include "DescriptorAllocator.h"

// Every slot of the one shader-visible CBV/SRV/UAV heap the whole frame binds.
// Enum order IS the heap layout: root-signature descriptor ranges are built from
// IndexOf(), so slots are appended at the end, never inserted.
enum class GlobalDescriptor : uint32_t
{
    ImGuiFont,
    CameraMatrices,
    RaytraceOutput,
    Tlas,
    Vertices,
    Indices,
    MaterialTextures, // Constants::Graphics::MAX_TEXTURES slots, one per glTF texture
    Skybox,
    VoxelOccupancy,
    VoxelIrradiance,
    VoxelVplCount,
    ShadingPoints,
    SuperpixelIndex,
    SuperpixelCenter,
    VoxelRepresentative,
    VplPosition,
    VBuffer,
    SpixelGathered,
    SpixelCounter,
    ClusterVisibilityMask,
    FuzzyWeight,
    FuzzyIndex,
    DebugViewOutput, // painted by DebugViewPass, copied to the back buffer
    Count
};

inline constexpr uint32_t GlobalDescriptorSlotCounts[] = {
    1,                                 // ImGuiFont
    1,                                 // CameraMatrices
    1,                                 // RaytraceOutput
    1,                                 // Tlas
    1,                                 // Vertices
    1,                                 // Indices
    Constants::Graphics::MAX_TEXTURES, // MaterialTextures
    1,                                 // Skybox
    1,                                 // VoxelOccupancy
    1,                                 // VoxelIrradiance
    1,                                 // VoxelVplCount
    1,                                 // ShadingPoints
    1,                                 // SuperpixelIndex
    1,                                 // SuperpixelCenter
    1,                                 // VoxelRepresentative
    1,                                 // VplPosition
    1,                                 // VBuffer
    1,                                 // SpixelGathered
    1,                                 // SpixelCounter
    1,                                 // ClusterVisibilityMask
    1,                                 // FuzzyWeight
    1,                                 // FuzzyIndex
    1,                                 // DebugViewOutput
};

static_assert(std::size(GlobalDescriptorSlotCounts) == static_cast<size_t>(GlobalDescriptor::Count), "Every GlobalDescriptor needs a slot count");

// ADR 0017 L2: the frame-global heap. Slot indices come from this layout instead
// of hand-chained constants, and slots whose resource dies get a null view rather
// than a dangling one.
class GlobalDescriptorHeap
{
public:
    static GlobalDescriptorHeap& Get();

    void Initialize(ID3D12Device* device);

    static constexpr uint32_t IndexOf(GlobalDescriptor slot)
    {
        uint32_t index = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(slot); ++i)
            index += GlobalDescriptorSlotCounts[i];
        return index;
    }

    static constexpr uint32_t Capacity() { return IndexOf(GlobalDescriptor::Count); }

    [[nodiscard]] ID3D12DescriptorHeap* GetHeap() const { return m_allocator.GetHeap(); }
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(GlobalDescriptor slot, uint32_t offset = 0) const;
    [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE GpuStart() const;

    // glTF material textures: one slot per texture out of the MaterialTextures
    // range. Returns -1 when the range is full (the caller keeps its old index).
    int AllocateMaterialTextureSlot();
    // Scene switch: the old scene's textures are about to be released, so every
    // slot they own goes back to the free list with a null view written over it.
    void ReleaseMaterialTextureSlots();

    // Writes the slot's null view — for producers that release a resource without
    // immediately creating its replacement.
    void ClearSlot(GlobalDescriptor slot);

private:
    DescriptorAllocator m_allocator;
    uint32_t            m_allocatedMaterialTextureSlots = 0;
    bool                m_initialized                   = false;
};
