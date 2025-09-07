#pragma once

#include "pch.h"

#include "Helpers.h"
#include "ResourceManager/ResourceManagerTypes.h"

struct Primitive
{
    explicit Primitive (const ResourceId id ) : id(id) {}
    
    ResourceId id;

    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;

    uint32_t firstVertex = 0;
    uint32_t firstIndex = 0;

    BYTE* vertexBufferCpu = nullptr;
    BYTE* indexBufferCpu = nullptr;

    Microsoft::WRL::ComPtr<ID3D12Resource> vertexBufferGpu = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> indexBufferGpu = nullptr;

    Microsoft::WRL::ComPtr<ID3D12Resource> vertexBufferUploader = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> indexBufferUploader = nullptr;

    UINT vertexByteStride = 0;
    UINT vertexBufferByteSize = 0;
    DXGI_FORMAT indexFormat = DXGI_FORMAT_R16_UINT;
    UINT indexBufferByteSize = 0;

    D3D12_VERTEX_BUFFER_VIEW GetVertexBufferView() const
    {
        D3D12_VERTEX_BUFFER_VIEW vbv{};
        vbv.BufferLocation = vertexBufferGpu->GetGPUVirtualAddress();
        vbv.StrideInBytes = vertexByteStride;
        vbv.SizeInBytes = vertexBufferByteSize;

        return vbv;
    }

    D3D12_INDEX_BUFFER_VIEW GetIndexBufferView() const
    {
        D3D12_INDEX_BUFFER_VIEW ibv{};
        ibv.BufferLocation = indexBufferGpu->GetGPUVirtualAddress();
        ibv.Format = indexFormat;
        ibv.SizeInBytes = indexBufferByteSize;

        return ibv;
    }

    void ReleaseResources()
    {
        AssertFreeClear(&vertexBufferCpu);
        AssertFreeClear(&indexBufferCpu);
        AssertReleaseClear(vertexBufferGpu);
        AssertReleaseClear(indexBufferGpu);
        AssertReleaseClear(vertexBufferUploader);
        AssertReleaseClear(indexBufferUploader);
        
        this->~Primitive();
    }
};
