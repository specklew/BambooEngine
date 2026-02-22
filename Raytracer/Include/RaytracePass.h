#pragma once
#include "Resources/StructuredBuffer.h"

class Renderer;
class Scene;
class ShaderBindingTable;
class AccelerationStructures;

class RaytracePass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<Scene> initialScene,
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvUavHeap,
        Microsoft::WRL::ComPtr<ID3D12Resource> randomBuffer);
    
    void Render(const Microsoft::WRL::ComPtr<ID3D12Resource>& renderTarget);
    void Update(double elapsedTime, double totalTime);
    void OnResize();
    void OnShaderReload();
    void OnSceneChange(std::shared_ptr<Scene> scene);
    
private:
    void InitializeRaytracingPipeline();

    void CreateRootSignatures();
    void CreateRayGenSignature();
    void CreateMissSignature();
    void CreateHitSignature();
    void CreateGlobalRootSignature();

    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
    void CreateShaderBindingTable();
    
    // Properties
    //std::vector<std::shared_ptr<AccelerationStructures>> m_accelerationStructures;
    
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

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_cbDescriptorHeap;

    std::shared_ptr<ShaderBindingTable> m_shaderBindingTable;
    
    std::shared_ptr<Scene> m_currentScene;

    D3D12_CPU_DESCRIPTOR_HANDLE m_geometryInfoHandle;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_randomBuffer;

    float m_time = 0.0f;
};
