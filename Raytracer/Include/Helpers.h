#pragma once
#include <comdef.h>

inline void Verify(HRESULT hr)
{
   if (!SUCCEEDED(hr))
   {
      spdlog::dump_backtrace();
      
      _com_error error(hr);
      LPCTSTR errMsg = error.ErrorMessage();
      std::wstring w;
      w = errMsg;
      std::string errorMessage = std::string(w.begin(), w.end()); // magic here
      spdlog::error("Verification of HR failed with code: {}", hr);
      spdlog::error("Error message: {}", errorMessage);
      
      assert(false);
   }
}

void ThrowIfFailed(HRESULT hr);