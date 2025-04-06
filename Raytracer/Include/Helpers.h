#pragma once

void ThrowIfFailed(HRESULT hr);
char* FormatTempString(const char* format, ...);
void ReadTextFromFile(const char* szFilepath, char* buffer, int bufferSize);
void ReadDataFromFile(const char* szFilepath, void* buffer, int bufferSize, bool text);
void WriteTextToFile(const char* szFilepath, const char* buffer, int bufferSize);
void WriteDataToFile(const char* szFilepath, const char* buffer, int bufferSize, const char* fileMode);

template <typename T>
__forceinline void AssertFreeClear(T** ptr)
{
    assert(*ptr != nullptr);
    free(*ptr);
    *ptr = nullptr;
}

namespace Math
{
    DirectX::XMFLOAT4X4 Identity4x4();
}
