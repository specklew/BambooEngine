#pragma once

void ThrowIfFailed(HRESULT hr);
std::string ConvertWcharToString(const wchar_t* wstr);
char* FormatTempString(const char* format, ...);

void ReadTextFromFile(const char* szFilepath, char* buffer, int bufferSize);
void ReadDataFromFile(const char* szFilepath, void* buffer, int bufferSize, bool text);
void WriteTextToFile(const char* szFilepath, const char* buffer, int bufferSize);
void WriteDataToFile(const char* szFilepath, const char* buffer, int bufferSize, const char* fileMode);

inline UINT64 Align(UINT64 size, UINT64 alignment)
{
    return (size + (alignment - 1)) & ~(alignment - 1);
}

std::string GetName(ID3D12Object *d3dObject);

template <typename T>
__forceinline void AssertReleaseClear(Microsoft::WRL::ComPtr<T>& ptr)
{
    assert(ptr != nullptr);
    ptr.Reset();
    ptr = nullptr;
}

template <typename T>
__forceinline void AssertDeleteSingleClear(T** ptr)
{
    assert(*ptr != nullptr);
    delete *ptr;
    *ptr = nullptr;
}

template <typename T>
__forceinline void AssertDeleteArrayClear(T** ptr)
{
    assert(*ptr != nullptr);
    delete[] *ptr;
    *ptr = nullptr;
}

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