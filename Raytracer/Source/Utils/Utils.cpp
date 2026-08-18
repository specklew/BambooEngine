#include "pch.h"
#include "CommandContext.h"
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
		CommandContext& context = CommandContext::Get();
		context.TransitionRaw(defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_COPY_DEST);
		// GetCommandList() flushes the queued barrier ahead of the copy.
		UpdateSubresources<1>(context.GetCommandList(), defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
		context.TransitionRaw(defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_GENERIC_READ);
		// Note: uploadBuffer has to be kept alive after the above function
		// calls because the command list has not been executed yet that
		// performs the actual copy.
		// The caller can Release the uploadBuffer after it knows the copy
		// has been executed.
		return defaultBuffer;
	}

	// Tightly packed RGBA8 copy of a glTF image, so mip generation has one layout to work on.
	static std::vector<uint8_t> ExpandToRgba8(const tinygltf::Image& image)
	{
		const size_t pixelCount = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
		std::vector<uint8_t> rgba(pixelCount * 4, 255);

		if (image.component == 4)
		{
			memcpy(rgba.data(), image.image.data(), pixelCount * 4);
		}
		else if (image.component == 3)
		{
			for (size_t px = 0; px < pixelCount; px++)
			{
				rgba[px * 4 + 0] = image.image[px * 3 + 0];
				rgba[px * 4 + 1] = image.image[px * 3 + 1];
				rgba[px * 4 + 2] = image.image[px * 3 + 2];
			}
		}
		else
		{
			spdlog::error("Unsupported image component count: {}", image.component);
		}

		return rgba;
	}

	// 2x2 box filter. An odd level clamps its last row/column onto itself instead of
	// reading past the level, which double-weights that one edge texel.
	static std::vector<uint8_t> DownsampleRgba8(const std::vector<uint8_t>& src, uint32_t srcWidth, uint32_t srcHeight,
		uint32_t dstWidth, uint32_t dstHeight)
	{
		std::vector<uint8_t> dst(static_cast<size_t>(dstWidth) * dstHeight * 4);

		for (uint32_t y = 0; y < dstHeight; y++)
		{
			const uint32_t y0 = std::min(y * 2, srcHeight - 1);
			const uint32_t y1 = std::min(y * 2 + 1, srcHeight - 1);

			for (uint32_t x = 0; x < dstWidth; x++)
			{
				const uint32_t x0 = std::min(x * 2, srcWidth - 1);
				const uint32_t x1 = std::min(x * 2 + 1, srcWidth - 1);

				for (uint32_t channel = 0; channel < 4; channel++)
				{
					const uint32_t sum = src[(static_cast<size_t>(y0) * srcWidth + x0) * 4 + channel]
					                   + src[(static_cast<size_t>(y0) * srcWidth + x1) * 4 + channel]
					                   + src[(static_cast<size_t>(y1) * srcWidth + x0) * 4 + channel]
					                   + src[(static_cast<size_t>(y1) * srcWidth + x1) * 4 + channel];
					dst[(static_cast<size_t>(y) * dstWidth + x) * 4 + channel] = static_cast<uint8_t>((sum + 2) / 4);
				}
			}
		}

		return dst;
	}

	ComPtr<ID3D12Resource> CreateDefaultTexture(
		ID3D12Device* device,
		ID3D12GraphicsCommandList* commandList,
		const tinygltf::Image& image,
		ComPtr<ID3D12Resource>& uploadBuffer)
    {
		spdlog::debug("Creating default texture resource.");

		// Full mip chain, built once at load. Without it every sampler is pinned to mip 0
		// and any pattern finer than a pixel aliases into large moire blocks - the
		// veach-ajar floor checker (2 texels per check in v) being the loud case.
		std::vector<std::vector<uint8_t>> mips{ ExpandToRgba8(image) };
		std::vector<std::pair<uint32_t, uint32_t>> mipSizes{
			{ static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height) } };
		while (mipSizes.back().first > 1 || mipSizes.back().second > 1)
		{
			const uint32_t srcWidth  = mipSizes.back().first;
			const uint32_t srcHeight = mipSizes.back().second;
			const uint32_t dstWidth  = std::max(1u, srcWidth / 2);
			const uint32_t dstHeight = std::max(1u, srcHeight / 2);

			mips.push_back(DownsampleRgba8(mips.back(), srcWidth, srcHeight, dstWidth, dstHeight));
			mipSizes.push_back({ dstWidth, dstHeight });
		}
		const UINT mipCount = static_cast<UINT>(mips.size());
    	
	    ComPtr<ID3D12Resource> defaultTexture;
	    {
		    const auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	    	const auto textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, image.width, image.height, 1, static_cast<UINT16>(mipCount));

	    	ThrowIfFailed(device->CreateCommittedResource(
				&heapProperties,
				D3D12_HEAP_FLAG_NONE,
				&textureDesc,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(&defaultTexture)));
	    }
    	
    	std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(mipCount);
    	std::vector<UINT> rowCounts(mipCount);
    	std::vector<UINT64> rowSizes(mipCount);
    	UINT64 size;
    	auto desc = defaultTexture->GetDesc();

    	device->GetCopyableFootprints(&desc, 0, mipCount, 0, footprints.data(), rowCounts.data(), rowSizes.data(), &size);
	    
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

    	for (UINT mip = 0; mip < mipCount; mip++)
    	{
    		const UINT srcRowPitch = mipSizes[mip].first * 4;
    		for (UINT row = 0; row < rowCounts[mip]; row++)
    		{
    			memcpy(
    				static_cast<uint8_t*>(pData) + footprints[mip].Offset
    					+ static_cast<UINT64>(footprints[mip].Footprint.RowPitch) * row,
    				mips[mip].data() + static_cast<size_t>(srcRowPitch) * row,
    				srcRowPitch);
    		}
    	}

    	for (UINT mip = 0; mip < mipCount; mip++)
    	{
    		D3D12_TEXTURE_COPY_LOCATION defaultCopyLocation = {};
    		defaultCopyLocation.pResource = defaultTexture.Get();
    		defaultCopyLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    		defaultCopyLocation.SubresourceIndex = mip;

    		D3D12_TEXTURE_COPY_LOCATION uploadCopyLocation = {};
    		uploadCopyLocation.pResource = uploadBuffer.Get();
    		uploadCopyLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    		uploadCopyLocation.PlacedFootprint = footprints[mip];

    		CommandContext::Get().GetCommandList()->CopyTextureRegion(&defaultCopyLocation, 0, 0, 0, &uploadCopyLocation, nullptr);
    	}

    	// PIXEL | NON_PIXEL (matches the skybox): scene textures are sampled by
    	// the raster PS, the RT passes AND compute kernels (inline-RayQuery
    	// integrator); a pixel-only state faults compute Dispatches under
    	// GPU-based validation (ADR 0003 cvis note — lifted by this widening).
    	CommandContext::Get().TransitionRaw(defaultTexture.Get(),
		    D3D12_RESOURCE_STATE_COPY_DEST,
		    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    	
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
