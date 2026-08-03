#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "RenderGraph.h"
#include "Resources/Texture.h"
#include "VxpgGraphHandles.h"

class PassConstants;
class Scene;

// What the frame offers a technique building its part of the graph. The display
// chain (accumulate -> tonemap -> present copy) belongs to the renderer, so a
// technique that renders offscreen hands its image back through BuildGraph's
// return value instead of reaching for the back buffer itself.
struct FrameGraphContext
{
    GraphResourceHandle backBuffer   = InvalidGraphResource;
    GraphResourceHandle depthStencil = InvalidGraphResource;

    // Render target / depth views of the two handles above.
    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRtv   = {};
    D3D12_CPU_DESCRIPTOR_HANDLE depthStencilDsv = {};

    // Never null; every handle is Invalid on a frame with no guiding subgraph.
    const VxpgGraphHandles* voxelGuiding = nullptr;
};

// A selectable way of producing the frame's image. Rasterization and every
// raytracing integrator are peers here: each declares its own nodes, and the
// graph decides what actually runs.
class RenderTechnique
{
public:
    virtual ~RenderTechnique() = default;

    virtual void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::shared_ptr<Scene> initialScene,
        std::shared_ptr<PassConstants> passConstants) = 0;

    // Adds this technique's nodes to the frame's graph. Returns the offscreen
    // image the renderer's display chain should consume, or InvalidGraphResource
    // when the technique has already drawn into the back buffer.
    virtual GraphResourceHandle BuildGraph(RenderGraph& graph, const FrameGraphContext& frame) = 0;

    // The resource behind that handle; null for a technique that draws straight
    // into the back buffer and therefore needs no display chain.
    virtual Texture* GetOutputTexture() const { return nullptr; }

    // Whether this technique samples the VXPG subgraph at all. Which of its
    // stages run is derived by the render graph from the reads the technique
    // declares, not from a stage order maintained here.
    virtual bool UsesVoxelGuiding() const { return false; }

    // PassConstants::debugMode carries a different enumeration depending on which
    // technique is active — raster views and raytracing views are separate lists
    // over the same field, so the technique that interprets it supplies it.
    // ADR 0017 step 4 merges the enums and this goes away.
    virtual int GetDebugMode() const { return 0; }

    // True when a debug view of this technique's own kind is selected, so the
    // shader variant carrying the view branches has to be compiled (ADR 0014).
    virtual bool HasActiveDebugView() const { return false; }

    virtual void OnResize() {}
    virtual void OnShaderReload() {}
    virtual void OnSceneChange(std::shared_ptr<Scene> scene) {}

    // Shader-variant switches (ADR 0014 / ADR 0015). Return true when the value
    // changed — the caller then owes a pipeline rebuild. A technique that
    // compiles no variants reports "nothing changed", so the renderer's sync
    // block needs no branch on what kind of technique is active.
    virtual bool SetDebugViewsCompiled(bool enabled) { return false; }
    virtual bool SetOneSampleMisCompiled(bool enabled) { return false; }

    // Registry — populated via REGISTER_TECHNIQUE before main().
    struct Entry
    {
        std::string name;
        std::function<std::shared_ptr<RenderTechnique>()> create;
    };
    static std::vector<Entry>& GetRegistry();
    static int RegisterTechnique(std::string name, std::function<std::shared_ptr<RenderTechnique>()> factory);
};

// Place at file scope in a .cpp alongside the subclass definition.
#define REGISTER_TECHNIQUE(Name, Class) \
    static int _reg_##Class = RenderTechnique::RegisterTechnique( \
        Name, []() { return std::static_pointer_cast<RenderTechnique>(std::make_shared<Class>()); });
