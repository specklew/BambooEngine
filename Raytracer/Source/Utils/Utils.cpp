#include "pch.h"
#include "Utils/Utils.h"
#include <comdef.h>

#include "Renderer.h"

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
			_com_error deviceRemovedErr(Renderer::g_d3d12Device->GetDeviceRemovedReason());
			LPCTSTR deviceRemovedMsg = deviceRemovedErr.ErrorMessage();
			std::wstring w2 = deviceRemovedMsg;
			std::string deviceRemovedError = std::string(w2.begin(), w2.end());
			spdlog::error("REASON: {}", deviceRemovedError);
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

DirectX::XMFLOAT4X4 MathUtils::XMFloat4x4Identity()
{
	DirectX::XMFLOAT4X4 result;
	XMStoreFloat4x4(&result, DirectX::XMMatrixIdentity());
	return result;
}

void MathUtils::PrintMatrix(const DirectX::XMFLOAT4X4& matrix)
{
	spdlog::info("Printing Matrix:");
	for (const auto row : matrix.m)
	{
		spdlog::info("| {:>8.4} {:>8.4} {:>8.4} {:>8.4} |",
		             row[0],
		             row[1],
		             row[2],
		             row[3]);
	}
}

void MathUtils::PrintMatrix(const DirectX::XMMATRIX& matrix)
{
	spdlog::info("Printing Matrix:");
	for (int i = 0; i < 4; i++)
	{
		spdlog::info("| {:>8.4} {:>8.4} {:>8.4} {:>8.4} |",
		             matrix.r[i].m128_f32[0],
		             matrix.r[i].m128_f32[1],
		             matrix.r[i].m128_f32[2],
		             matrix.r[i].m128_f32[3]);
	}
}
