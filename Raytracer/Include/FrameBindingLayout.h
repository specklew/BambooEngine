#pragma once
#include <vector>

#include "FrameBindingRegisters.h"

class Scene;
class PassConstants;

// C++ mirror of Resources/Shaders/FrameBindings.hlsl. Both files declare the same
// register layout; shader reflection validates them against each other every time a
// PSO is built, so a drift between the two is a logged error rather than a garbage
// read (ADR 0017, phase 4).
//
// space0 holds the frame resources every raytracing pass sees. space1 is reserved
// for per-pass bindings and must stay empty here. Everything in this layout is
// bound for every pass regardless of use, so the list stays short by hand.
namespace FrameBindingLayout
{
    // Typed views of FrameBindingRegisters.h, which HLSL reads from the same file.
    // t-registers
    inline constexpr uint32_t kTlas              = FRAME_REG_TLAS;
    inline constexpr uint32_t kVertices          = FRAME_REG_VERTICES;
    inline constexpr uint32_t kIndices           = FRAME_REG_INDICES;
    inline constexpr uint32_t kGeometryInfo      = FRAME_REG_GEOMETRY_INFO;
    inline constexpr uint32_t kInstanceInfo      = FRAME_REG_INSTANCE_INFO;
    inline constexpr uint32_t kEmissiveTriangles = FRAME_REG_EMISSIVE_TRIANGLES;
    inline constexpr uint32_t kLightData         = FRAME_REG_LIGHT_DATA;
    inline constexpr uint32_t kLightPool         = FRAME_REG_LIGHT_POOL;
    inline constexpr uint32_t kSkybox            = FRAME_REG_SKYBOX;
    inline constexpr uint32_t kMaterialTextures  = FRAME_REG_MATERIAL_TEXTURES; // …+ MAX_TEXTURES - 1

    // u-registers
    inline constexpr uint32_t kRaytraceOutput = FRAME_REG_RAYTRACE_OUTPUT;

    // b-registers
    inline constexpr uint32_t kCameraMatrices = FRAME_REG_CAMERA_MATRICES;
    inline constexpr uint32_t kPassConstants  = FRAME_REG_PASS_CONSTANTS;

    // Root parameter slots. Frame parameters occupy a fixed prefix in every
    // raytracing root signature so one bind call can serve all of them; a pass
    // appends its own parameters starting at kPassRootParameterStart.
    enum RootParameter : uint32_t
    {
        FrameTable = 0,
        GeometryInfoSrv,
        InstanceInfoSrv,
        LightDataSrv,
        EmissiveTrianglesSrv,
        LightPoolSrv,
        PassConstantsCbv,
        kPassRootParameterStart
    };

    // Descriptor ranges of the frame table, in FrameTable's parameter. A pass
    // appends its own ranges to the same vector before creating the signature.
    void AppendFrameRanges(std::vector<D3D12_DESCRIPTOR_RANGE>& ranges);

    // Whether a register belongs to the frame layout. Frame bindings are present
    // in every signature whether the pass reads them or not, so the unused-binding
    // report has to skip them or it reports nothing else.
    [[nodiscard]] bool IsFrameRegister(D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t baseRegister,
                                       uint32_t registerSpace);

    // Fills rootParameters[FrameTable … PassConstantsCbv]. The ranges vector must
    // outlive the signature's serialization.
    void FillFrameRootParameters(CD3DX12_ROOT_PARAMETER*                   rootParameters,
                                 const std::vector<D3D12_DESCRIPTOR_RANGE>& ranges);

    // Binds the frame prefix on the compute root signature. Every raytracing pass
    // calls this instead of repeating six near-identical binds.
    void BindFrameRootParameters(ID3D12GraphicsCommandList* commandList, Scene& scene,
                                 const PassConstants& passConstants);
}
