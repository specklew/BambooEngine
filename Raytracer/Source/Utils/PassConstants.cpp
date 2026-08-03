#include "pch.h"
#include "Utils/PassConstants.h"

#include "Renderer.h"

PassConstants::PassConstants()
{
    m_ring.Initialize(Renderer::g_device.Get(), sizeof(MappedData), L"Pass Constants");
}

void PassConstants::Map(uint32_t frameIndex)
{
    m_frameIndex = frameIndex;
    m_ring.Write(frameIndex, &data, sizeof(MappedData));
}

D3D12_GPU_VIRTUAL_ADDRESS PassConstants::GetGpuVirtualAddress() const
{
    return m_ring.GetGpuVirtualAddress(m_frameIndex);
}
