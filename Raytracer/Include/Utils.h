#pragma once

namespace RenderingUtils
{
    Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* cmdList,
        const void* initData,
        UINT64 byteSize,
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