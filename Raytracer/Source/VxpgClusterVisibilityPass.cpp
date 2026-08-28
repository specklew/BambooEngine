#include "pch.h"
#include "CommandContext.h"
#include "VxpgClusterVisibilityPass.h"

#include "Constants.h"
#include "GlobalDescriptorHeap.h"
#include "Renderer.h" // GetStaticSamplers
#include "VoxelizationPass.h"
#include "VoxelGuidingBuildPass.h"
#include "VxpgClusterPass.h"
#include "SuperpixelBuildPass.h"
#include "SceneResources/Scene.h"
#include "PassRegisters.h"
#include "ResourceManager/ResourceManager.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "Utils/CameraConstants.h"
#include "Utils/CVars.h"
#include "Utils/Utils.h"

using Microsoft::WRL::ComPtr;

namespace
{
// It includes RaytracingUtils.hlsl, so the scene half of the layout is frame
// registers; the rest is cvis-specific. No texture range: the BSDF weight uses
// per-instance material factors, so this compute pass never samples the scene
// textures (which sit in the raster path's PIXEL_SHADER_RESOURCE layout,
// illegal for a compute Dispatch).
constexpr BindingSlot kCvisTlas     = TableEntry("SceneBVH", BindingKind::Srv, FRAME_REG_TLAS, GlobalDescriptor::Tlas);
constexpr BindingSlot kCvisVertices = TableEntry("g_vertices", BindingKind::Srv, FRAME_REG_VERTICES, GlobalDescriptor::Vertices);
constexpr BindingSlot kCvisIndices  = TableEntry("g_indices", BindingKind::Srv, FRAME_REG_INDICES, GlobalDescriptor::Indices);
constexpr BindingSlot kCvisSuperpixelIndex =
    PassTableEntry("gSuperpixelIndex", BindingKind::Uav, CVIS_REG_SUPERPIXEL_INDEX, GlobalDescriptor::SuperpixelIndex);
constexpr BindingSlot kCvisVplPosition =
    PassTableEntry("gVplPosition", BindingKind::Uav, CVIS_REG_VPL_POSITION, GlobalDescriptor::VplPosition);
constexpr BindingSlot kCvisVBuffer = PassTableEntry("gVBuffer", BindingKind::Uav, CVIS_REG_VBUFFER, GlobalDescriptor::VBuffer);
constexpr BindingSlot kCvisSpixelGathered =
    PassTableEntry("gSpixelGathered", BindingKind::Uav, CVIS_REG_SPIXEL_GATHERED, GlobalDescriptor::SpixelGathered);
constexpr BindingSlot kCvisSpixelCounter =
    PassTableEntry("gSpixelCounter", BindingKind::Uav, CVIS_REG_SPIXEL_COUNTER, GlobalDescriptor::SpixelCounter);
constexpr BindingSlot kCvisVisibilityMask = PassTableEntry("gClusterVisibilityMask", BindingKind::Uav,
                                                       CVIS_REG_VISIBILITY_MASK, GlobalDescriptor::ClusterVisibilityMask);

constexpr BindingSlot kCvisCamera       = RootCbv("CameraParams", FRAME_REG_CAMERA_MATRICES);
constexpr BindingSlot kCvisGeometryInfo = RootSrv("g_geometryInfo", FRAME_REG_GEOMETRY_INFO);
constexpr BindingSlot kCvisInstanceInfo = RootSrv("g_instanceInfo", FRAME_REG_INSTANCE_INFO);
constexpr BindingSlot kCvisGridConstants   = PassCbv("CvisGridCB", CVIS_REG_GRID_CB);
constexpr BindingSlot kCvisConstants       = PassRootConstants("CvisCB", CVIS_REG_CB, 7);
constexpr BindingSlot kCvisInverseIndex    = PassUav("gVoxInverseIndex", CVIS_REG_INVERSE_INDEX);
constexpr BindingSlot kCvisAssignments     = PassUav("gVoxelClusterAssignments", CVIS_REG_CLUSTER_ASSIGNMENTS);
constexpr BindingSlot kCvisGatheredPoints  = PassUav("gClusterGatheredLightPoints", CVIS_REG_GATHERED_LIGHT_POINTS);
constexpr BindingSlot kCvisPointCounts     = PassUav("gClusterLightPointCounts", CVIS_REG_LIGHT_POINT_COUNTS);
constexpr BindingSlot kCvisAvgVisibility   = PassUav("gSpixelClusterAvgVisibility", CVIS_REG_AVG_VISIBILITY);

constexpr BindingSlot kCvisSlots[] = {
    kCvisTlas,            kCvisVertices,       kCvisIndices,        kCvisSuperpixelIndex,
    kCvisVplPosition,     kCvisVBuffer,        kCvisSpixelGathered, kCvisSpixelCounter, kCvisVisibilityMask,
    kCvisCamera,          kCvisGeometryInfo,   kCvisInstanceInfo,   kCvisGridConstants, kCvisConstants,
    kCvisInverseIndex,    kCvisAssignments,    kCvisGatheredPoints, kCvisPointCounts,   kCvisAvgVisibility};
} // namespace

// SIByL cvis defaults: use_bsdf = true, use_distance = false (BRDF-weighted soft
// visibility). Both exposed so the weighting can be toggled for measurement.
static AutoCVarInt g_cvisUseBsdf("vxpg.cvis.useBsdf",
    "Weight cluster visibility by the receiver Cook-Torrance BRDF (SIByL default on)",
    1, CVarFlags::EditCheckbox);

namespace
{
    constexpr uint32_t kClusterCount = 32;
    constexpr uint32_t kGatherCap = 1024;
    constexpr uint32_t kSuperpixelSize = Constants::Graphics::SUPERPIXEL_SIZE;
}

void VxpgClusterVisibilityPass::Initialize(
    ComPtr<ID3D12Device5>              device,
    ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::shared_ptr<VoxelizationPass>      voxelPass,
    std::shared_ptr<VoxelGuidingBuildPass> buildPass,
    std::shared_ptr<VxpgClusterPass>       clusterPass,
    std::shared_ptr<SuperpixelBuildPass>   superpixelPass)
{
    spdlog::info("Initializing VXPG cluster-visibility pass...");

    m_device         = device;
    m_commandList    = commandList;
    m_voxelPass      = std::move(voxelPass);
    m_buildPass      = std::move(buildPass);
    m_clusterPass    = std::move(clusterPass);
    m_superpixelPass = std::move(superpixelPass);

    CreateFixedBuffers();
    CreateRootSignature();
    CreatePSOs();

    m_initialized = true;
}

void VxpgClusterVisibilityPass::CreateFixedBuffers()
{
    m_clusterGatheredLightPoints = std::make_unique<RWStructuredBuffer<DirectX::XMFLOAT4>>(
        m_device, kClusterCount * kGatherCap, L"ClusterVisibility GatheredLightPoints");
    m_clusterLightPointCounts = std::make_unique<RWStructuredBuffer<uint32_t>>(
        m_device, kClusterCount, L"ClusterVisibility LightPointCounts");
}

void VxpgClusterVisibilityPass::CreateResolutionBuffers()
{
    m_avgVisibility = std::make_unique<RWStructuredBuffer<float>>(
        m_device, std::max(1u, m_mapX * m_mapY * kClusterCount), L"ClusterVisibility AvgVisibility");

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32_UINT, std::max(1u, m_mapX), std::max(1u, m_mapY),
        1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_mask)));
    m_mask->SetName(L"ClusterVisibility Mask");
}

void VxpgClusterVisibilityPass::CreateRootSignature()
{
    // No texture range: the BSDF weight uses per-instance material factors, so
    // this compute pass never samples the scene textures. (Scene textures are
    // created PIXEL | NON_PIXEL, so a compute Dispatch could legally read them —
    // the reason is cost, not state; see the DEVIATION note in the shader.)
    // It includes RaytracingUtils.hlsl, so the scene half of its bindings is the
    // frame layout; only the cvis-specific UAVs are pass-scoped.
    m_rootSig = RootSignatureBuilder(L"VxpgClusterVisibility RootSig", /*tableCount*/ 1)
                    .Add(kCvisSlots)
                    .WithStaticSamplers()
                    .Build(m_device.Get());
}

void VxpgClusterVisibilityPass::CreatePSOs()
{
    auto& cache = ShaderProgramCache::Get();

    m_clearProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgClusterVisibility.clear.shader", L"Cvis Clear PSO");
    m_gatherProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgClusterVisibility.gather.shader", L"Cvis Gather PSO");
    m_checkProgram = cache.GetOrCreateCompute(m_device.Get(), m_rootSig.Get(),
        "resources/shaders/vxpgClusterVisibility.check.shader", L"Cvis Check PSO");
}

void VxpgClusterVisibilityPass::OnResize(uint32_t width, uint32_t height)
{
    m_width  = width;
    m_height = height;
    m_mapX   = (width  + kSuperpixelSize - 1) / kSuperpixelSize;
    m_mapY   = (height + kSuperpixelSize - 1) / kSuperpixelSize;
    CreateResolutionBuffers();
}

bool VxpgClusterVisibilityPass::BindCommon(uint32_t frameIndex)
{
    if (!m_initialized || !m_scene || !m_voxelPass || !m_buildPass ||
        !m_clusterPass || !m_superpixelPass || m_mapX == 0)
        return false;

    auto* cmd = m_commandList.Get();

    ID3D12DescriptorHeap* heaps[] = { GlobalDescriptorHeap::Get().GetHeap() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);

    cmd->SetComputeRootSignature(m_rootSig.Get());
    m_rootSig.SetTable(cmd, 0, GlobalDescriptorHeap::Get().GpuStart());
    m_rootSig.Set(cmd, kCvisCamera, CameraConstants::Get().GetGpuVirtualAddress());
    m_rootSig.Set(cmd, kCvisGeometryInfo, m_scene->GetGeometryInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kCvisInstanceInfo, m_scene->GetInstanceInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kCvisGridConstants, m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());

    uint32_t constants[7] = {
        m_width, m_height, m_mapX, m_mapY, frameIndex,
        static_cast<uint32_t>(g_cvisUseBsdf.Get() != 0),
        m_scene->GetInstanceInfoBuffer()->GetElementsCount()
    };
    m_rootSig.SetConstants(cmd, kCvisConstants, constants, 7);
    m_rootSig.Set(cmd, kCvisInverseIndex, m_buildPass->GetInverseIndexBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kCvisAssignments, m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kCvisGatheredPoints, m_clusterGatheredLightPoints->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kCvisPointCounts, m_clusterLightPointCounts->GetGPUVirtualAddress());
    m_rootSig.Set(cmd, kCvisAvgVisibility, m_avgVisibility->GetGPUVirtualAddress());

    return true;
}

void VxpgClusterVisibilityPass::RunClear(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_clearProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_mapX + 15) / 16, (m_mapY + 15) / 16, 1);
}

// File each pixel's VPL into its cluster drawer and seed the mask bits for the
// connections it already proves.
void VxpgClusterVisibilityPass::RunGather(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_gatherProgram->GetPipelineState());
    CommandContext::Get().Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);
}

// Shadow-ray probes -> soft avg-visibility + mask. The dispatch covers mapX
// superpixels wide (32 sample lanes each) and mapY*4 groups tall (8 clusters per
// group x 4 = 32 clusters per superpixel row).
void VxpgClusterVisibilityPass::RunCheck(uint32_t frameIndex)
{
    if (!BindCommon(frameIndex))
        return;

    m_commandList->SetPipelineState(m_checkProgram->GetPipelineState());
    CommandContext::Get().Dispatch(m_mapX, m_mapY * 4, 1);
}
