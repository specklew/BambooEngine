#include "pch.h"
#include "Resources/Buffer.h"

Buffer::Buffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc)
    :   Resource(device, desc) {}

Buffer::Buffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource)
    : Resource(device, resource) {}
