#include "pch.h"
#include "Utils/Utils.h"
#include <comdef.h>

#include "Renderer.h"
#include "tinygltf/tiny_gltf.h"

// DRED post-mortem: which command in which command list the GPU died on, plus
// the page-faulting allocation if the reason was an invalid address.
static void DumpDeviceRemovedExtendedData()
{
	Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData> dred;
	if (FAILED(Renderer::g_device->QueryInterface(IID_PPV_ARGS(&dred))))
	{
		spdlog::error("DRED: not available (QueryInterface failed)");
		return;
	}

	D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs = {};
	if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs)))
	{
		for (const D3D12_AUTO_BREADCRUMB_NODE* node = breadcrumbs.pHeadAutoBreadcrumbNode;
		     node != nullptr; node = node->pNext)
		{
			const UINT last = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
			if (last == 0 || last == node->BreadcrumbCount)
				continue; // fully executed or never started -> not the culprit

			const char* listName = node->pCommandListDebugNameA ? node->pCommandListDebugNameA : "<unnamed list>";
			spdlog::error("DRED: command list '{}' IN FLIGHT at breadcrumb {}/{}",
				listName, last, node->BreadcrumbCount);
			const UINT from = (last > 8) ? last - 8 : 0;
			const UINT to = (last + 4 < node->BreadcrumbCount) ? last + 4 : node->BreadcrumbCount;
			for (UINT i = from; i < to; ++i)
				spdlog::error("DRED:   op[{}] = {}{}", i,
					static_cast<uint32_t>(node->pCommandHistory[i]),
					(i == last) ? "   <-- LAST EXECUTED (6=Dispatch 15=Barrier 34=DispatchRays 3/4=Draw 31=BuildRTAS)" : "");
		}
	}

	D3D12_DRED_PAGE_FAULT_OUTPUT pageFault = {};
	if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&pageFault)) && pageFault.PageFaultVA != 0)
	{
		spdlog::error("DRED: PAGE FAULT at GPU VA 0x{:X}", pageFault.PageFaultVA);
		// Bamboo names resources with wide SetName -> names land in ObjectNameW.
		auto nodeName = [](const D3D12_DRED_ALLOCATION_NODE* node) -> std::string
		{
			if (node->ObjectNameW) return ConvertWcharToString(node->ObjectNameW);
			if (node->ObjectNameA) return node->ObjectNameA;
			return "<unnamed>";
		};
		for (const D3D12_DRED_ALLOCATION_NODE* node = pageFault.pHeadExistingAllocationNode;
		     node != nullptr; node = node->pNext)
			spdlog::error("DRED:   existing allocation: '{}' (type {})", nodeName(node),
				static_cast<uint32_t>(node->AllocationType));
		for (const D3D12_DRED_ALLOCATION_NODE* node = pageFault.pHeadRecentFreedAllocationNode;
		     node != nullptr; node = node->pNext)
			spdlog::error("DRED:   recently freed: '{}' (type {})", nodeName(node),
				static_cast<uint32_t>(node->AllocationType));
	}
	else
	{
		spdlog::error("DRED: no page fault recorded (likely a hang/TDR, not a bad address)");
	}

	spdlog::default_logger()->flush(); // process dies right after; don't lose this
}

void ThrowIfFailed(HRESULT hr)
{
	if (FAILED(hr))
	{
		spdlog::dump_backtrace();

		_com_error error(hr);
		LPCTSTR errMsg = error.ErrorMessage();
		std::wstring w;
		w = errMsg;
		std::string errorMessage = std::string(w.begin(), w.end()); // magic here
		spdlog::error("Verification of HR failed with code: {}", hr);
		spdlog::error("Error message: {}", errorMessage);

		if (hr == 2289696773)
		{
			//DEVICE REMOVED
			spdlog::error("ERROR caused by DEVICE REMOVED. TRANSLATING...");
			_com_error deviceRemovedErr(Renderer::g_device->GetDeviceRemovedReason());
			LPCTSTR deviceRemovedMsg = deviceRemovedErr.ErrorMessage();
			std::wstring w2 = deviceRemovedMsg;
			std::string deviceRemovedError = std::string(w2.begin(), w2.end());
			spdlog::error("REASON: {}", deviceRemovedError);
			DumpDeviceRemovedExtendedData();
		}

   		throw std::runtime_error(errorMessage);
	}
}

std::string ConvertWcharToString(const wchar_t* wstr)
{
   std::wstring ws(wstr);
   std::string str(ws.begin(), ws.end());
   return str;
}

std::string ToLowerAscii(std::string s)
{
   for (char& c : s)
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
   return s;
}

char* FormatTempString(const char* format, ...)
{
   static char buffer[4096];

   va_list args;
   va_start(args, format);
   vsprintf_s(buffer, format, args);
   va_end(args);

   return buffer;
}

void ReadTextFromFile(const char* szFilepath, char* buffer, int bufferSize)
{
   ReadDataFromFile(szFilepath, buffer, bufferSize, true);
}

void ReadDataFromFile(const char* szFilepath, void* buffer, int bufferSize, bool text)
{
   static_assert(sizeof(char) == 1);
   assert(szFilepath && buffer && bufferSize > 0);

   FILE* file = nullptr;
   const errno_t err = fopen_s(&file, szFilepath, text? "r" : "rb");
   if (err != 0 || file == nullptr)
   {
      SPDLOG_ERROR("failed to open file for reading: {}", szFilepath);
      assert(false && "Failed to open file for reading. See console for details");
   }
   const size_t bytesRead = fread_s(buffer, bufferSize, sizeof(char), bufferSize, file);
   const int ret = fclose(file);
   assert(ret == 0);
   assert(ferror(file) == 0);
   assert(bytesRead > 0);
   if (text)
   {
      // make sure there is space for null terminator character, hence <
      assert(bytesRead < bufferSize && "Not enough space in the given buffer to read the whole file");
      static_cast<char*>(buffer)[bytesRead] = '\0';
   }
   else
   {
      // for binary data we do not append anything, hence can use the entire buffer.
      assert(bytesRead <= bufferSize && "Not enough space in the given buffer to read the whole file");
   }
}

void WriteTextToFile(const char* szFilepath, const char* buffer, int bufferSize)
{
   WriteDataToFile(szFilepath, buffer, bufferSize, "w"); // text mode
}

void WriteDataToFile(const char* szFilepath, const char* buffer, int bufferSize, const char* fileMode)
{
   static_assert(sizeof(char) == 1);
   assert(szFilepath && buffer && bufferSize > 0);

   FILE* file = nullptr;
   const errno_t err = fopen_s(&file, szFilepath, fileMode);
   assert(err == 0 && file != nullptr && "Failed to open file for writing");
   const size_t bytesWritten = fwrite(buffer, sizeof(char), bufferSize, file);
   const int ret = fclose(file);
   assert(ret == 0);
   assert(bytesWritten == bufferSize);
   assert(ferror(file) == 0);
}



std::string GetName(ID3D12Object *d3dObject)
{
   wchar_t name[128] = { };
   uint32_t name_size = sizeof(name);
   if (SUCCEEDED(d3dObject->GetPrivateData(WKPDID_D3DDebugObjectNameW, &name_size, name)))
   {
      return ConvertWcharToString(name);
   }
   
   return "Unnamed D3D12 Object";
}

namespace RenderingUtils
{
    using namespace DirectX;
    using namespace Microsoft::WRL;

    ComPtr<ID3D12Resource> CreateDefaultBuffer(
		ID3D12Device* device,
		ID3D12GraphicsCommandList* commandList,
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
			commandList->ResourceBarrier(1, &barrier);
		}
		UpdateSubresources<1>(commandList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
		{
			const auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
				D3D12_RESOURCE_STATE_COPY_DEST,
				D3D12_RESOURCE_STATE_GENERIC_READ);
			commandList->ResourceBarrier(1, &barrier);
		}
		// Note: uploadBuffer has to be kept alive after the above function
		// calls because the command list has not been executed yet that
		// performs the actual copy.
		// The caller can Release the uploadBuffer after it knows the copy
		// has been executed.
		return defaultBuffer;
	}

	ComPtr<ID3D12Resource> CreateDefaultTexture(
		ID3D12Device* device,
		ID3D12GraphicsCommandList* commandList,
		const tinygltf::Image& image,
		ComPtr<ID3D12Resource>& uploadBuffer)
    {
		spdlog::debug("Creating default texture resource.");
    	
	    ComPtr<ID3D12Resource> defaultTexture;
	    {
		    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	    	const auto textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, image.width, image.height, 1, 1);

	    	ThrowIfFailed(device->CreateCommittedResource(
				&heapProperties,
				D3D12_HEAP_FLAG_NONE,
				&textureDesc,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(&defaultTexture)));
	    }
    	
    	D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    	UINT rowCount;
    	UINT64 rowSize;
    	UINT64 size;
    	auto desc = defaultTexture->GetDesc();

    	device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rowCount, &rowSize, &size);
	    
	    {
		    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	    	const auto textureDesc = CD3DX12_RESOURCE_DESC::Buffer(size);

	    	ThrowIfFailed(device->CreateCommittedResource(
	    		&heapProperties,
	    		D3D12_HEAP_FLAG_NONE,
	    		&textureDesc,
	    		D3D12_RESOURCE_STATE_GENERIC_READ,
	    		nullptr,
	    		IID_PPV_ARGS(&uploadBuffer)));
	    }

    	void* pData;
    	ThrowIfFailed(uploadBuffer->Map(0, nullptr, &pData));

    	const UINT dstRowPitch = footprint.Footprint.RowPitch;

    	if (image.component == 4)
    	{
    		for (UINT i = 0; i < rowCount; i++)
    		{
    			memcpy(
    				static_cast<uint8_t*>(pData) + dstRowPitch * i,
    				&image.image[0] + image.width * 4 * i,
    				image.width * 4);
    		}
    	}
    	else if (image.component == 3)
    	{
    		for (UINT i = 0; i < rowCount; i++)
    		{
    			const uint8_t* srcRow = &image.image[0] + image.width * 3 * i;
    			uint8_t* dstRow = static_cast<uint8_t*>(pData) + dstRowPitch * i;
    			for (int px = 0; px < image.width; px++)
    			{
    				dstRow[px * 4 + 0] = srcRow[px * 3 + 0];
    				dstRow[px * 4 + 1] = srcRow[px * 3 + 1];
    				dstRow[px * 4 + 2] = srcRow[px * 3 + 2];
    				dstRow[px * 4 + 3] = 255;
    			}
    		}
    	}
    	else
    	{
    		spdlog::error("Unsupported image component count: {}", image.component);
    	}

    	D3D12_TEXTURE_COPY_LOCATION defaultCopyLocation = {};
    	defaultCopyLocation.pResource = defaultTexture.Get();
    	defaultCopyLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    	defaultCopyLocation.SubresourceIndex = 0;

    	D3D12_TEXTURE_COPY_LOCATION uploadCopyLocation = {};
    	uploadCopyLocation.pResource = uploadBuffer.Get();
    	uploadCopyLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    	uploadCopyLocation.PlacedFootprint = footprint;

    	commandList->CopyTextureRegion(&defaultCopyLocation, 0, 0, 0, &uploadCopyLocation, nullptr);

    	// PIXEL | NON_PIXEL (matches the skybox): scene textures are sampled by
    	// the raster PS, the RT passes AND compute kernels (inline-RayQuery
    	// integrator); a pixel-only state faults compute Dispatches under
    	// GPU-based validation (ADR 0003 cvis note — lifted by this widening).
    	auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
		    defaultTexture.Get(),
		    D3D12_RESOURCE_STATE_COPY_DEST,
		    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		commandList->ResourceBarrier(1, &barrier);
    	
    	return defaultTexture;
    }

    ComPtr<ID3D12Resource> CreateUavBuffer(
        ID3D12Device* device,
        UINT64 byteSize,
        const wchar_t* name)
    {
        byteSize = Align(byteSize, 256);

        const auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        ComPtr<ID3D12Resource> buffer;
        HRESULT hr = device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&buffer));
        if (FAILED(hr))
        {
            spdlog::error("CreateUavBuffer failed (size={}, hr={:#010x}, deviceRemoved={:#010x})",
                byteSize, static_cast<uint32_t>(hr),
                static_cast<uint32_t>(device->GetDeviceRemovedReason()));
            ThrowIfFailed(hr);
        }
        if (name)
            buffer->SetName(name);
        return buffer;
    }
}

DirectX::XMFLOAT4X4 MathUtils::XMFloat4x4Identity()
{
	DirectX::XMFLOAT4X4 result;
	XMStoreFloat4x4(&result, DirectX::XMMatrixIdentity());
	return result;
}

void MathUtils::PrintMatrix(const DirectX::XMFLOAT4X4& matrix)
{
	spdlog::debug("ix:");
	for (const auto row : matrix.m)
	{
		spdlog::debug("| {:>8.4} {:>8.4} {:>8.4} {:>8.4} |",
		             row[0],
		             row[1],
		             row[2],
		             row[3]);
	}
}

void MathUtils::PrintMatrix(const DirectX::XMMATRIX& matrix)
{
	spdlog::debug("ix:");
	for (int i = 0; i < 4; i++)
	{
		spdlog::debug("| {:>8.4} {:>8.4} {:>8.4} {:>8.4} |",
		             matrix.r[i].m128_f32[0],
		             matrix.r[i].m128_f32[1],
		             matrix.r[i].m128_f32[2],
		             matrix.r[i].m128_f32[3]);
	}
}
