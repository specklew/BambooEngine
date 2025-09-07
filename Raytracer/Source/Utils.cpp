﻿#include "pch.h"
#include "Utils.h"

namespace RenderingUtils
{
    using namespace DirectX;
    using namespace Microsoft::WRL;

    ComPtr<ID3D12Resource> CreateDefaultBuffer(
		ID3D12Device* device,
		ID3D12GraphicsCommandList* cmdList,
		const void* initData,
		const UINT64 byteSize,
		ComPtr<ID3D12Resource>& uploadBuffer)
	{
		ComPtr<ID3D12Resource> defaultBuffer;
		// Create the actual default buffer resource.
		const auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		const auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);

		HRESULT hr = device->CreateCommittedResource(
			&defaultHeap,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			IID_PPV_ARGS(defaultBuffer.GetAddressOf()));

    	assert(SUCCEEDED(hr));
    	
		// In order to copy CPU memory data into our default buffer, we need
		// to create an intermediate upload heap.
		const auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    	
		hr = device->CreateCommittedResource(
			&uploadHeap,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(uploadBuffer.GetAddressOf()));

    	assert(SUCCEEDED(hr));
    	
		// Describe the data we want to copy into the default buffer.
		D3D12_SUBRESOURCE_DATA subResourceData = {};
		subResourceData.pData = initData;
		subResourceData.RowPitch = byteSize;
		subResourceData.SlicePitch = byteSize;
		// Schedule to copy the data to the default buffer resource.
		// At a high level, the helper function UpdateSubresources
		// will copy the CPU memory into the intermediate upload heap.
		// Then, using ID3D12CommandList::CopySubresourceRegion,
		// the intermediate upload heap data will be copied to mBuffer.
		{
			const auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_COPY_DEST);
			cmdList->ResourceBarrier(1, &barrier);
		}
		UpdateSubresources<1>(cmdList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
		{
			const auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
				D3D12_RESOURCE_STATE_COPY_DEST,
				D3D12_RESOURCE_STATE_GENERIC_READ);
			cmdList->ResourceBarrier(1, &barrier);
		}
		// Note: uploadBuffer has to be kept alive after the above function
		// calls because the command list has not been executed yet that
		// performs the actual copy.
		// The caller can Release the uploadBuffer after it knows the copy
		// has been executed.
		return defaultBuffer;
	}
}