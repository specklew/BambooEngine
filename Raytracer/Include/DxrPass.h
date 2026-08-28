#pragma once
#include "CommandContext.h"
#include "Resources/StructuredBuffer.h"
#include "Resources/Texture.h"
#include "RootSignatureLibrary.h"
#include "Techniques/TechniqueDescriptor.h"

class PassConstants;
class Scene;
class ShaderBindingTable;
class AccelerationStructures;

// A DXR dispatch and everything it takes to build one: shader libraries, hit
// groups, local and global root signatures, the state object and the SBT. Not a
// technique — auxiliary passes (VBuffer, light injection) are DXR dispatches the
// frame needs but the user never selects, so they derive from this directly.
// A selectable integrator derives from DxrTechnique instead.
class DxrPass
{
public:
    virtual ~DxrPass() = default;

    virtual void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<Scene> initialScene,
        std::shared_ptr<PassConstants> passConstants);

    virtual void Render();

    virtual void OnResize();
    virtual void OnShaderReload();
    virtual void OnSceneChange(std::shared_ptr<Scene> scene);

protected:
    // Subclasses override this to define their shaders, hit groups, and pipeline config.
    // Default implementation reproduces the original path tracing setup.
    virtual TechniqueDesc GetTechniqueDesc() const;

    // Override to customize local root signatures (default: one empty sig per role group).
    virtual void CreateLocalRootSignatures();
    // Override to customize the global root signature (default: the FrameBindingLayout
    // scene binding — one descriptor table plus seven root descriptors).
    virtual void CreateGlobalRootSignature();

    // Checks every library's reflected bindings against the global root signature.
    void ValidateShaderBindings() const;

    // Pipeline build — iterates TechniqueDesc returned by GetTechniqueDesc().
    // Override only if you need a fundamentally different pipeline structure.
    virtual void InitializeRaytracingPipeline();
    virtual void CreateShaderBindingTable();

    // Descriptor writes into the shared heap. The base writes the TLAS every DXR
    // pass needs; an override adds its own views and calls up. Also the resize
    // hook: anything created here is recreated at the new render dimensions.
    virtual void CreateShaderResourceHeap();

    // Shared device/command interfaces
    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;

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
    RootSignature m_globalRootSignature;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_cbDescriptorHeap;

    std::shared_ptr<ShaderBindingTable> m_shaderBindingTable;
    std::shared_ptr<Scene>              m_currentScene;

    // Which compile-time vendor levers this pipeline was BUILT with (ADR 0020).
    // Frozen at build time so a CVar flipped mid-frame cannot desync the state
    // object from the blobs it was made of; the Renderer compares it against the
    // live key and asks for a rebuild when they differ. Empty = the plain shaders.
    std::string m_shaderVariantKey;

    bool CompilesLever(const char* leverName) const
    {
        return m_shaderVariantKey.find(leverName) != std::string::npos;
    }

    // Launch grid for the integrator dispatch. Equals the image except under the
    // "swizzle" lever, whose Morton remap is a bijection only over whole tiles —
    // a short grid there would drop pixels and shade others twice (ADR 0020 R2).
    void GetLaunchExtent(uint32_t& width, uint32_t& height) const;

    D3D12_CPU_DESCRIPTOR_HANDLE m_geometryInfoHandle = {};

    std::shared_ptr<PassConstants> m_passConstants;

private:
    void LoadShaders();
};
