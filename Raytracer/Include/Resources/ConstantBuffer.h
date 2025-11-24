#pragma once
#include "Buffer.h"

class ConstantBuffer : public Buffer
{
public:
    ConstantBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource);

    void MapDataToWholeBuffer(BYTE* data) const;

    void Unmap() const;

    [[nodiscard]] size_t GetSizeInBytes() const { return m_sizeInBytes; }
    
private:
    size_t m_sizeInBytes;
};
