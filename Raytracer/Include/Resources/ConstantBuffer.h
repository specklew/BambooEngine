#pragma once
#include "Buffer.h"

class ConstantBuffer : public Buffer
{
public:
    ConstantBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource);

    void MapDataToWholeBuffer(void** data) const;
    void Unmap() const;
};
