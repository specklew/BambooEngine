#pragma once
#include <vector>

#include "BindingSlot.h"
#include "FrameBindingRegisters.h"

class PassConstants;
class RootSignature;
class Scene;

// C++ mirror of Resources/Shaders/FrameBindings.hlsl. Register numbers come from
// FrameBindingRegisters.h, which both files include, so the two cannot disagree
// (ADR 0019). What lives here is the rest of the layout — storage kind, heap
// slot, descriptor count — which is a C++-only concern.
//
// space0 holds the frame resources every raytracing pass sees. space1 is reserved
// for per-pass bindings and must stay empty here. Everything in this layout is
// bound for every pass regardless of use, so the list stays short by hand.
namespace FrameBindingLayout
{
    // Table entries, in table order.
    inline constexpr BindingSlot kRaytraceOutput = TableEntry("gOutput", BindingKind::Uav, FRAME_REG_RAYTRACE_OUTPUT, GlobalDescriptor::RaytraceOutput);
    inline constexpr BindingSlot kTlas           = TableEntry("SceneBVH", BindingKind::Srv, FRAME_REG_TLAS, GlobalDescriptor::Tlas);
    inline constexpr BindingSlot kVertices       = TableEntry("g_vertices", BindingKind::Srv, FRAME_REG_VERTICES, GlobalDescriptor::Vertices);
    inline constexpr BindingSlot kIndices        = TableEntry("g_indices", BindingKind::Srv, FRAME_REG_INDICES, GlobalDescriptor::Indices);
    inline constexpr BindingSlot kSkybox         = TableEntry("g_skybox", BindingKind::Srv, FRAME_REG_SKYBOX, GlobalDescriptor::Skybox);
    inline constexpr BindingSlot kMaterialTextures =
        TableEntry("g_textures", BindingKind::Srv, FRAME_REG_MATERIAL_TEXTURES, GlobalDescriptor::MaterialTextures, FRAME_MAX_TEXTURES);

    // Root descriptors, in root parameter order. The camera matrices are here
    // rather than in the table because they are written every frame: a root CBV
    // is bound by address, so each frame binds its own copy of the ring, which a
    // single heap descriptor could not express (see CameraConstants).
    inline constexpr BindingSlot kCameraMatrices    = RootCbv("CameraParams", FRAME_REG_CAMERA_MATRICES);
    inline constexpr BindingSlot kGeometryInfo      = RootSrv("g_geometryInfo", FRAME_REG_GEOMETRY_INFO);
    inline constexpr BindingSlot kInstanceInfo      = RootSrv("g_instanceInfo", FRAME_REG_INSTANCE_INFO);
    inline constexpr BindingSlot kLightData         = RootSrv("g_lightData", FRAME_REG_LIGHT_DATA);
    inline constexpr BindingSlot kEmissiveTriangles = RootSrv("g_emissiveTriangles", FRAME_REG_EMISSIVE_TRIANGLES);
    inline constexpr BindingSlot kLightPool         = RootSrv("g_lightPool", FRAME_REG_LIGHT_POOL);
    inline constexpr BindingSlot kPassConstants     = RootCbv("PassConstants", FRAME_REG_PASS_CONSTANTS);

    // The layout, in the order every raytracing signature declares it. One list:
    // the signature is built from it, membership is answered from it, and the
    // per-frame bind walks it.
    const std::vector<BindingSlot>& Slots();

    // The six static samplers, named as the shaders declare them. They are part
    // of the signature even though they are not root parameters, so validation
    // needs them or every shader that samples looks like it reads an undeclared
    // register. Added by RootSignatureBuilder::WithStaticSamplers().
    const std::vector<BindingSlot>& StaticSamplerSlots();

    // Whether a register belongs to the frame layout. Frame bindings are present
    // in every signature whether the pass reads them or not, so the unused-binding
    // report has to skip them or it reports nothing else.
    [[nodiscard]] bool IsFrameRegister(uint32_t registerSpace);

    // Binds the whole frame layout. Every raytracing pass calls this instead of
    // repeating seven near-identical binds.
    void Bind(ID3D12GraphicsCommandList* commandList, const RootSignature& rootSignature, Scene& scene,
              const PassConstants& passConstants);
}
