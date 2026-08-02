#pragma once
#include "Resources/StructuredBuffer.h"
#include "Resources/Texture.h"
#include "Techniques/TechniqueDescriptor.h"
#include "RasterDebugMode.h"

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
        std::shared_ptr<PassConstants> passConstants);

    virtual void Render();

    // Furthest VXPG pipeline stage this technique needs run before it dispatches.
    // Default None: most techniques (vanilla path tracing, AO) use no VXPG data.
    // Whether this technique samples the VXPG subgraph at all. Which of its
    // stages run is derived by the render graph from the reads the technique
    // declares, not from a stage order maintained here.
    virtual bool UsesVoxelGuiding() const { return false; }

    void OnResize();
    void OnShaderReload();
    void OnSceneChange(std::shared_ptr<Scene> scene);

    // Debug-view shader variant switch: true compiles the raygen with the
    // debug-view branches, false picks the clean benchmark variant. Returns
    // true when the value changed — caller then owes a pipeline rebuild
    // (Renderer::OnShaderReload; GPU must be idle before the state object swap).
    bool SetDebugViewsCompiled(bool enabled);

    // One-sample MIS raygen variant (ADR 0015): compiled only while
    // vxpg.oneSampleMis is on — the estimator code must not ride in (and
    // codegen-tax) the faithful two-sample build. Same rebuild contract as
    // SetDebugViewsCompiled. Only the guided technique's GetTechniqueDesc
    // reads the flag.
    bool SetOneSampleMisCompiled(bool enabled);

    // Null for auxiliary passes that override CreateRaytracingOutputBuffer empty.
    Texture* GetOutputTexture() const { return m_outputResource.get(); }

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

    // Checks every library's reflected bindings against the global root signature.
    void ValidateShaderBindings() const;

    // Pipeline build — iterates TechniqueDesc returned by GetTechniqueDesc().
    // Override only if you need a fundamentally different pipeline structure.
    virtual void InitializeRaytracingPipeline();
    virtual void CreateShaderBindingTable();

    // Output buffer + heap descriptor writes — auxiliary passes without their own
    // full-screen output (e.g. light injection) override these to skip clobbering
    // the shared heap's output UAV slot.
    virtual void CreateRaytracingOutputBuffer();
    virtual void CreateShaderResourceHeap();

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
    std::unique_ptr<Texture>                     m_outputResource;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_cbDescriptorHeap;

    std::shared_ptr<ShaderBindingTable> m_shaderBindingTable;
    std::shared_ptr<Scene>              m_currentScene;

    // Default true = views-in raygen, matching renderer.raygenCleanVariant's
    // default 0 so startup skips a rebuild. Views-in measured FASTER on the
    // current RDNA driver (Renderer variant-sync block has the numbers);
    // GetTechniqueDesc overrides pick the rg sidecar off this flag.
    bool m_compileDebugViews = true;
    bool m_compileOneSampleMis = false; // matches vxpg.oneSampleMis default 0

    D3D12_CPU_DESCRIPTOR_HANDLE m_geometryInfoHandle = {};

    std::shared_ptr<PassConstants> m_passConstants;

private:
    void LoadShaders();
};
