#include "pch.h"
#include "CommandContext.h"
#include "DebugViewPass.h"

#include "GlobalDescriptorHeap.h"
#include "PassRegisters.h"
#include "RootSignatureLibrary.h"
#include "Shader.h"
#include "VoxelGuidingBuildPass.h"
#include "VoxelizationPass.h"
#include "VxpgClusterPass.h"
#include "Window.h"
#include "ResourceManager/ResourceManager.h"

namespace
{
// Everything rides the global heap, because every resource this paints from is
// already published there by the VXPG passes. The output joins them so the whole
// pass is one table plus two constant buffers.
constexpr BindingSlot kDebugOutput = Accesses(
    PassTableEntry("gDebugOutput", BindingKind::Uav, DEBUG_VIEW_REG_OUTPUT, GlobalDescriptor::DebugViewOutput),
    GraphAccess::ComputeWrite);
constexpr BindingSlot kShadingPoints = Accesses(
    PassTableEntry("gShadingPoints", BindingKind::Uav, DEBUG_VIEW_REG_SHADING_POINTS, GlobalDescriptor::ShadingPoints),
    GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kVoxelOccupancy = Accesses(
    PassTableEntry("gVoxelOccupancy", BindingKind::Uav, DEBUG_VIEW_REG_VOXEL_OCCUPANCY, GlobalDescriptor::VoxelOccupancy),
    GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kVoxelIrradiance = Accesses(
    PassTableEntry("gVoxelIrradiance", BindingKind::Uav, DEBUG_VIEW_REG_VOXEL_IRRADIANCE, GlobalDescriptor::VoxelIrradiance),
    GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kVoxelVplCount = Accesses(
    PassTableEntry("gVoxelVplCount", BindingKind::Uav, DEBUG_VIEW_REG_VOXEL_VPL_COUNT, GlobalDescriptor::VoxelVplCount),
    GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kSuperpixelIndex = Accesses(
    PassTableEntry("gSuperpixelIndex", BindingKind::Uav, DEBUG_VIEW_REG_SUPERPIXEL_INDEX, GlobalDescriptor::SuperpixelIndex),
    GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kSuperpixelCenter = Accesses(
    PassTableEntry("gSuperpixelCenter", BindingKind::Uav, DEBUG_VIEW_REG_SUPERPIXEL_CENTER, GlobalDescriptor::SuperpixelCenter),
    GraphAccess::UnorderedAccessRead);

// The cluster products are structured buffers rather than global-heap views, so
// they bind as root UAVs — the same way the guided technique reads them.
constexpr BindingSlot kInverseIndex = Accesses(
    PassUav("gVoxInverseIndex", DEBUG_VIEW_REG_INVERSE_INDEX), GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kCounters = Accesses(
    PassUav("gVoxCounters", DEBUG_VIEW_REG_COUNTERS), GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kClusterAssignments = Accesses(
    PassUav("gVoxelClusterAssignments", DEBUG_VIEW_REG_CLUSTER_ASSIGN), GraphAccess::UnorderedAccessRead);
constexpr BindingSlot kClusterSeeds = Accesses(
    PassUav("gClusterSeedCompactIds", DEBUG_VIEW_REG_CLUSTER_SEEDS), GraphAccess::UnorderedAccessRead);

constexpr BindingSlot kVoxelGridConstants = PassCbv("VoxelGridCB", REG_VOXEL_GRID_CB);
constexpr BindingSlot kDebugViewConstants = PassRootConstants("DebugViewCB", DEBUG_VIEW_REG_CB, 4);

constexpr BindingSlot kDebugViewSlots[] = {
    kDebugOutput,     kShadingPoints,    kVoxelOccupancy,      kVoxelIrradiance,
    kVoxelVplCount,   kSuperpixelIndex,  kSuperpixelCenter,    kInverseIndex,
    kCounters,        kClusterAssignments, kClusterSeeds,      kVoxelGridConstants,
    kDebugViewConstants};
} // namespace

void DebugViewPass::Initialize(Microsoft::WRL::ComPtr<ID3D12Device5> device,
                               Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    m_device      = device;
    m_commandList = commandList;

    CreateResources();
    CreateRootSignature();
}

void DebugViewPass::CreateRootSignature()
{
    m_rootSignature = RootSignatureBuilder(L"DebugViewPass RootSig", /*tableCount*/ 1)
                          .Add(kDebugViewSlots)
                          .Build(m_device.Get());
}

void DebugViewPass::CreateResources()
{
    m_outputBuffer.reset();

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM; // matches the back buffer, so the present is a straight copy
    desc.Width            = Window::Get().GetWidth();
    desc.Height           = Window::Get().GetHeight();
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    m_outputBuffer = std::make_unique<Texture>(m_device, desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Debug View Output");

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_outputBuffer->GetUnderlyingResource().Get(), nullptr, &uavDesc,
        GlobalDescriptorHeap::Get().CpuHandle(GlobalDescriptor::DebugViewOutput));
}

void DebugViewPass::OnResize()
{
    CreateResources();
}

void DebugViewPass::DeclareGraphResources(RenderGraphPassBuilder& pass, GraphResourceHandle output,
                                          const VxpgGraphHandles& vxpg) const
{
    pass.Declare(kDebugOutput, output);

    // Only the buffers the active view actually samples. Declaring the whole
    // table would keep the entire VXPG chain alive for a view that reads one
    // texture — the point of a per-view declaration is that culling drops the
    // stages nothing behind them needs.
    switch (g_bufferDebugView.Get())
    {
    case BufferDebugView::VoxelOccupancy:
        pass.Declare(kShadingPoints, vxpg.shadingPoints);
        pass.Declare(kVoxelOccupancy, vxpg.voxelOccupancy);
        break;
    // Keeps the whole world-space chain alive through culling — fingerprint,
    // cluster seeding and assignment all have to run for this view to say
    // anything.
    case BufferDebugView::VoxelClusters:
        pass.Declare(kShadingPoints, vxpg.shadingPoints);
        pass.Declare(kInverseIndex, vxpg.inverseIndex);
        pass.Declare(kCounters, vxpg.counters);
        pass.Declare(kClusterAssignments, vxpg.clusterAssignments);
        pass.Declare(kClusterSeeds, vxpg.clusterSeedCompactIds);
        break;
    case BufferDebugView::VoxelIrradiance:
        pass.Declare(kShadingPoints, vxpg.shadingPoints);
        pass.Declare(kVoxelIrradiance, vxpg.voxelIrradiance);
        pass.Declare(kVoxelVplCount, vxpg.voxelVplCount);
        break;
    case BufferDebugView::ShadingPointsNormal:
    case BufferDebugView::ShadingPointsPos:
        pass.Declare(kShadingPoints, vxpg.shadingPoints);
        break;
    case BufferDebugView::SuperpixelId:
        pass.Declare(kSuperpixelIndex, vxpg.superpixelIndex);
        break;
    case BufferDebugView::SuperpixelRepresentative:
        pass.Declare(kSuperpixelIndex, vxpg.superpixelIndex);
        pass.Declare(kSuperpixelCenter, vxpg.superpixelCenter);
        break;
    case BufferDebugView::None:
        break;
    }
}

void DebugViewPass::Dispatch()
{
    if (!m_program)
        m_program = ShaderProgramCache::Get().GetOrCreateCompute(m_device.Get(), m_rootSignature.Get(),
            "resources/shaders/debugViewPaint.cs.shader", L"DebugViewPaint PSO");

    ID3D12DescriptorHeap* heaps[] = {GlobalDescriptorHeap::Get().GetHeap()};
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_rootSignature.SetTable(m_commandList.Get(), 0, GlobalDescriptorHeap::Get().GpuStart());

    if (m_voxelPass)
        m_rootSignature.Set(m_commandList.Get(), kVoxelGridConstants,
                            m_voxelPass->GetGridConstantsBuffer()->GetGPUVirtualAddress());

    // Root descriptors are not bounds-checked, so a view that does not read them
    // must still be given something legal to point at. Only the cluster view
    // declares them, and it cannot run without both producers present.
    if (m_buildPass && m_clusterPass)
    {
        m_rootSignature.Set(m_commandList.Get(), kInverseIndex,
                            m_buildPass->GetInverseIndexBuffer()->GetGPUVirtualAddress());
        m_rootSignature.Set(m_commandList.Get(), kCounters,
                            m_buildPass->GetCountersBuffer()->GetGPUVirtualAddress());
        m_rootSignature.Set(m_commandList.Get(), kClusterAssignments,
                            m_clusterPass->GetVoxelClusterAssignmentsBuffer()->GetGPUVirtualAddress());
        m_rootSignature.Set(m_commandList.Get(), kClusterSeeds,
                            m_clusterPass->GetClusterSeedCompactIdsBuffer()->GetGPUVirtualAddress());
    }

    const uint32_t width  = Window::Get().GetWidth();
    const uint32_t height = Window::Get().GetHeight();
    const uint32_t constants[4] = {static_cast<uint32_t>(g_bufferDebugView.Get()), width, height, 0u};
    m_rootSignature.SetConstants(m_commandList.Get(), kDebugViewConstants, constants, 4);

    m_commandList->SetPipelineState(m_program->GetPipelineState());
    CommandContext::Get().Dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

void DebugViewPass::CopyToBackBuffer(Texture& backBuffer)
{
    CommandContext::Get().CopyResource(
        backBuffer.GetUnderlyingResource().Get(), m_outputBuffer->GetUnderlyingResource().Get());
}
