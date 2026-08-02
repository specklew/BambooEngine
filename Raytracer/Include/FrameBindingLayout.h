#pragma once
#include <vector>

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
    // t-registers
    inline constexpr uint32_t kTlas              = 0;
    inline constexpr uint32_t kVertices          = 1;
    inline constexpr uint32_t kIndices           = 2;
    inline constexpr uint32_t kGeometryInfo      = 3;
    inline constexpr uint32_t kInstanceInfo      = 4;
    inline constexpr uint32_t kEmissiveTriangles = 5;
    inline constexpr uint32_t kLightData         = 6;
    inline constexpr uint32_t kLightPool         = 7;
    inline constexpr uint32_t kSkybox            = 8;
    inline constexpr uint32_t kMaterialTextures  = 16; // …16 + MAX_TEXTURES - 1

    // u-registers
    inline constexpr uint32_t kRaytraceOutput = 0;

    // b-registers
    inline constexpr uint32_t kCameraMatrices = 0;
    inline constexpr uint32_t kPassConstants  = 3;

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
