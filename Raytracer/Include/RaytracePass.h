#pragma once

namespace nv_helpers_dx12
{
    class ShaderBindingTableGenerator;
}

class AccelerationStructures;

class RaytracePass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList, 
        std::shared_ptr<AccelerationStructures> accelerationStructures,
        Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer,
        Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer);
    
    void Render(const Microsoft::WRL::ComPtr<ID3D12Resource>& renderTarget);
    void Update(DirectX::XMMATRIX view, DirectX::XMMATRIX proj);
    
private:
    void InitializeRaytracingPipeline();
    
    Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateRayGenSignature();
    Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateMissSignature();
    Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateHitSignature();
    void CreateGloablRootSignature();

    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
    void CreateShaderBindingTable();

    void CreateConstantCameraBuffer();
    
    // Properties
    std::shared_ptr<AccelerationStructures> m_accelerationStructures;
    
    Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;

    Microsoft::WRL::ComPtr<IDxcBlob> m_rayGenShaderBlob;
    Microsoft::WRL::ComPtr<IDxcBlob> m_missShaderBlob;
    Microsoft::WRL::ComPtr<IDxcBlob> m_hitShaderBlob;
    std::wstring m_rayGenShaderName;
    std::wstring m_missShaderName;
    std::wstring m_hitShaderName;
    std::wstring m_hitGroupName;

    Microsoft::WRL::ComPtr<ID3D12StateObject> m_rtStateObject;
    Microsoft::WRL::ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProperties;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rayGenSignature;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_missSignature;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_hitSignature;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_globalRootSignature;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_outputResource;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;

    std::shared_ptr<nv_helpers_dx12::ShaderBindingTableGenerator> m_shaderBindingTableGenerator = {};
    Microsoft::WRL::ComPtr<ID3D12Resource> m_shaderBindingTableStorage;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_cbDescriptorHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cbCamera;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBuffer;
};
