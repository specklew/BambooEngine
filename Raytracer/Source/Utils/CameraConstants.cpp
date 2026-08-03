#include "pch.h"
#include "Utils/CameraConstants.h"

CameraConstants& CameraConstants::Get()
{
    static CameraConstants s_cameraConstants;
    return s_cameraConstants;
}

void CameraConstants::Initialize(ID3D12Device* device)
{
    m_ring.Initialize(device, sizeof(MappedData), L"Camera Matrices");
}

void CameraConstants::Update(uint32_t frameIndex, const MappedData& matrices)
{
    m_frameIndex = frameIndex;
    m_ring.Write(frameIndex, &matrices, sizeof(matrices));
}

D3D12_GPU_VIRTUAL_ADDRESS CameraConstants::GetGpuVirtualAddress() const
{
    return m_ring.GetGpuVirtualAddress(m_frameIndex);
}
