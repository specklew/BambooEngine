#pragma once
#include "tinygltf/tiny_gltf.h"

void ThrowIfFailed(HRESULT hr);
std::string ConvertWcharToString(const wchar_t* wstr);
char* FormatTempString(const char* format, ...);

void ReadTextFromFile(const char* szFilepath, char* buffer, int bufferSize);
void ReadDataFromFile(const char* szFilepath, void* buffer, int bufferSize, bool text);
void WriteTextToFile(const char* szFilepath, const char* buffer, int bufferSize);
void WriteDataToFile(const char* szFilepath, const char* buffer, int bufferSize, const char* fileMode);

namespace RenderingUtils
{
    Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* commandList,
        const void* initData,
        UINT64 byteSize,
        Microsoft::WRL::ComPtr<ID3D12Resource>& uploadBuffer);

    Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultTexture(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* commandList,
        const tinygltf::Image& image,
        Microsoft::WRL::ComPtr<ID3D12Resource>& uploadBuffer);
}

namespace MathUtils
{
    DirectX::XMFLOAT4X4 XMFloat4x4Identity();
    /*DirectX::XMFLOAT4X4 XMFloat4x4Translation(float x, float y, float z);
    DirectX::XMFLOAT4X4 XMFloat4x4Scaling(float sx, float sy, float sz);
    DirectX::XMFLOAT4X4 XMFloat4x4RotationQuaternion(const DirectX::XMFLOAT4& quat);
    DirectX::XMFLOAT4X4 XMFloat4x4Multiply(const DirectX::XMFLOAT4X4& a, const DirectX::XMFLOAT4X4& b);*/
    void PrintMatrix(const DirectX::XMFLOAT4X4& matrix);
    void PrintMatrix(const DirectX::XMMATRIX& matrix);
}


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