#pragma once
#include "Resources/StructuredBuffer.h"
#include "Techniques/TechniqueDescriptor.h"

class PassConstants;
class Renderer;
class Scene;
class ShaderBindingTable;
class AccelerationStructures;

class RaytracePass
{
public:
    virtual ~RaytracePass() = default;

    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<Scene> initialScene,
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvUavHeap,
        Microsoft::WRL::ComPtr<ID3D12Resource> randomBuffer,
        std::shared_ptr<PassConstants> passConstants);

    virtual void Render();
    void Update(double elapsedTime, double totalTime);
    void OnResize();
    void OnShaderReload();
    void OnSceneChange(std::shared_ptr<Scene> scene);

    const Microsoft::WRL::ComPtr<ID3D12Resource>& GetOutputResource() const { return m_outputResource; }

    // Technique registry — populated via REGISTER_RAYTRACE_TECHNIQUE macro
    static std::vector<TechniqueEntry>& GetRegistry();
    static int RegisterTechnique(const std::string& name, std::function<std::shared_ptr<RaytracePass>()> factory);

protected:
    // Subclasses override this to define their shaders, hit groups, and pipeline config.
    // Default implementation reproduces the original path tracing setup.
    virtual TechniqueDesc GetTechniqueDesc() const;

    // Override to customize local root signatures (default: one empty sig per role group).
    virtual void CreateLocalRootSignatures();
    // Override to customize the global root signature (default: standard 7-param scene binding).
    virtual void CreateGlobalRootSignature();

    // Pipeline build — iterates TechniqueDesc returned by GetTechniqueDesc().
    // Override only if you need a fundamentally different pipeline structure.
    virtual void InitializeRaytracingPipeline();
    virtual void CreateShaderBindingTable();

    // Shared device/command interfaces
    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;

    // Compiled shader blobs — parallel to m_techniqueDesc.shaders, populated in CreateRootSignatures()
    std::vector<Microsoft::WRL::ComPtr<IDxcBlob>>      m_shaderBlobs;

    // Cached descriptor from GetTechniqueDesc(), set in InitializeRaytracingPipeline()
    TechniqueDesc m_techniqueDesc;

    // Pipeline state
    Microsoft::WRL::ComPtr<ID3D12StateObject>           m_rtStateObject;
    Microsoft::WRL::ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProperties;

    // Local root signatures (one per role group; empty by default, overridable)
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rayGenLocalSig;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_missLocalSig;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_hitLocalSig;

    // Global root signature
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_globalRootSignature;

    // Output resources
    Microsoft::WRL::ComPtr<ID3D12Resource>       m_outputResource;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_cbDescriptorHeap;

    std::shared_ptr<ShaderBindingTable> m_shaderBindingTable;
    std::shared_ptr<Scene>              m_currentScene;

    D3D12_CPU_DESCRIPTOR_HANDLE m_geometryInfoHandle = {};

    Microsoft::WRL::ComPtr<ID3D12Resource> m_randomBuffer;
    std::shared_ptr<PassConstants>         m_passConstants;

    float m_time = 0.0f;

private:
    void LoadShaders();
    void CreateRaytracingOutputBuffer();
    void CreateShaderResourceHeap();
};
