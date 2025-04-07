#include "pch.h"

#include "Helpers.h"
#include <comdef.h>

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

DirectX::XMFLOAT4X4 Math::Identity4x4()
{
   static DirectX::XMFLOAT4X4 I(
               1.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 1.0f, 0.0f, 0.0f,
               0.0f, 0.0f, 1.0f, 0.0f,
               0.0f, 0.0f, 0.0f, 1.0f);

   return I;
}


