#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "DebugViewDoc.h"
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
class GpuMemoryReport;

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

    // Every debug view this technique understands, "off" first. Headless walks
    // this to capture each view unattended, which is the only way a view gets
    // checked without a human at the window; the UI shows `doc` as a tooltip so
    // the expected image is readable at the point of choosing it.
    struct DebugView { int index; std::string name; std::string doc; };

    // The shared buffer views live above this, so a technique's own enumeration
    // and the technique-independent one can share a list without colliding.
    static constexpr int kBufferViewIndexBase = 1000;

    virtual std::vector<DebugView> GetDebugViews() const { return { {0, "None", ""} }; }

    // Selects one of the views above. Indices belong to this technique's own
    // enumeration, so a value from another technique's list is rejected.
    virtual bool SetDebugView(int index) { return index == 0; }

    // Appends the buffer views every technique can show (ADR 0017 phase 5b) to a
    // technique's own list, and routes a pick back to whichever owns it.
    static std::vector<DebugView> WithBufferViews(std::vector<DebugView> own);
    static bool SelectDebugView(RenderTechnique& technique, int index);

    // P5: a technique that holds GPU resources of its own declares them here. Empty by
    // default because path tracing holds none beyond its output, which is shared.
    virtual void ReportMemory(GpuMemoryReport& report) const {}
    virtual void OnResize() {}
    virtual void OnShaderReload() {}
    virtual void OnSceneChange(std::shared_ptr<Scene> scene) {}

    // Which compile-time vendor levers this technique should be built with
    // (ADR 0020's registry produces the key). Returns true when it changed — the
    // caller then owes a pipeline rebuild. A technique that compiles no variants
    // reports "nothing changed", so the renderer's sync block needs no branch on
    // what kind of technique is active.
    virtual bool SetShaderVariantKey(const std::string& key) { return false; }
    // What the live pipeline was actually BUILT with. Recorded next to every
    // capture: the lever CVars say what was asked for, this says what compiled,
    // and a measurement needs the second one.
    virtual std::string GetShaderVariantKey() const { return {}; }

    // Registry — populated via REGISTER_TECHNIQUE before main().
    struct Entry
    {
        std::string name;
        std::function<std::shared_ptr<RenderTechnique>()> create;
    };
    static std::vector<Entry>& GetRegistry();
    static int RegisterTechnique(std::string name, std::function<std::shared_ptr<RenderTechnique>()> factory);
};

// Turns a debug-view enum plus its doc table into the list the UI and headless
// walk. The docs array is already required to be one-per-entry in enum order
// (FormatDebugViewDocs static_asserts the count), so the two line up by index.
template <typename EnumType, size_t N>
std::vector<RenderTechnique::DebugView> BuildDebugViews(const DebugViewDoc (&docs)[N])
{
    static_assert(N == magic_enum::enum_count<EnumType>(), "every debug view needs a DebugViewDoc entry");

    std::vector<RenderTechnique::DebugView> views;
    views.reserve(N);
    size_t index = 0;
    for (const EnumType value : magic_enum::enum_values<EnumType>())
    {
        views.push_back({static_cast<int>(value), std::string(magic_enum::enum_name(value)),
                         FormatDebugViewDoc(docs[index])});
        ++index;
    }
    return views;
}

// Place at file scope in a .cpp alongside the subclass definition.
#define REGISTER_TECHNIQUE(Name, Class) \
    static int _reg_##Class = RenderTechnique::RegisterTechnique( \
        Name, []() { return std::static_pointer_cast<RenderTechnique>(std::make_shared<Class>()); });
