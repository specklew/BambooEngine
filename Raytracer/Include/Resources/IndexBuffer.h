#pragma once
#include "Buffer.h"

class IndexBuffer : public Buffer
{
public:
    size_t GetIndexCount() const { return m_indexCount; }
    DXGI_FORMAT GetIndexFormat() const { return m_indexFormat; }
    D3D12_INDEX_BUFFER_VIEW GetIndexBufferView() const { return m_indexBufferView; }
    
    IndexBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, size_t indexCount, DXGI_FORMAT format);
    IndexBuffer(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource>& resource, size_t indexCount, DXGI_FORMAT format);
    
private:
    D3D12_INDEX_BUFFER_VIEW CreateAndGetIndexBufferView() const;
    
    size_t m_indexCount;
    DXGI_FORMAT m_indexFormat;
    D3D12_INDEX_BUFFER_VIEW m_indexBufferView;
};
