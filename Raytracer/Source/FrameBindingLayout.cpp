#include "pch.h"
#include "FrameBindingLayout.h"

#include "Constants.h"
#include "GlobalDescriptorHeap.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"

namespace
{
    D3D12_DESCRIPTOR_RANGE MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t baseRegister,
                                     uint32_t count, GlobalDescriptor heapSlot)
    {
        D3D12_DESCRIPTOR_RANGE range;
        range.RangeType                         = type;
        range.BaseShaderRegister                = baseRegister;
        range.NumDescriptors                    = count;
        range.RegisterSpace                     = 0;
        range.OffsetInDescriptorsFromTableStart = GlobalDescriptorHeap::IndexOf(heapSlot);
        return range;
    }
}

void FrameBindingLayout::AppendFrameRanges(std::vector<D3D12_DESCRIPTOR_RANGE>& ranges)
{
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, kCameraMatrices, 1,
                               GlobalDescriptor::CameraMatrices));
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kRaytraceOutput, 1,
                               GlobalDescriptor::RaytraceOutput));
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kTlas, 1, GlobalDescriptor::Tlas));
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kVertices, 1, GlobalDescriptor::Vertices));
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kIndices, 1, GlobalDescriptor::Indices));
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kSkybox, 1, GlobalDescriptor::Skybox));
    ranges.push_back(MakeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kMaterialTextures,
                               Constants::Graphics::MAX_TEXTURES, GlobalDescriptor::MaterialTextures));
}

bool FrameBindingLayout::IsFrameRegister(D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t baseRegister,
                                         uint32_t registerSpace)
{
    if (registerSpace != 0)
        return false;

    switch (type)
    {
    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
        return baseRegister == kCameraMatrices || baseRegister == kPassConstants;
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
        return baseRegister == kRaytraceOutput;
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
        return baseRegister <= kSkybox ||
               (baseRegister >= kMaterialTextures &&
                baseRegister < kMaterialTextures + Constants::Graphics::MAX_TEXTURES);
    default:
        return false;
    }
}

void FrameBindingLayout::FillFrameRootParameters(CD3DX12_ROOT_PARAMETER*                   rootParameters,
                                                 const std::vector<D3D12_DESCRIPTOR_RANGE>& ranges)
{
    rootParameters[FrameTable].InitAsDescriptorTable(static_cast<UINT>(ranges.size()), ranges.data());
    rootParameters[GeometryInfoSrv].InitAsShaderResourceView(kGeometryInfo, 0);
    rootParameters[InstanceInfoSrv].InitAsShaderResourceView(kInstanceInfo, 0);
    rootParameters[LightDataSrv].InitAsShaderResourceView(kLightData, 0);
    rootParameters[EmissiveTrianglesSrv].InitAsShaderResourceView(kEmissiveTriangles, 0);
    rootParameters[LightPoolSrv].InitAsShaderResourceView(kLightPool, 0);
    rootParameters[PassConstantsCbv].InitAsConstantBufferView(kPassConstants, 0);
}

void FrameBindingLayout::BindFrameRootParameters(ID3D12GraphicsCommandList* commandList, Scene& scene,
                                                 const PassConstants& passConstants)
{
    ID3D12DescriptorHeap* heaps[] = {GlobalDescriptorHeap::Get().GetHeap()};
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);

    commandList->SetComputeRootDescriptorTable(FrameTable, GlobalDescriptorHeap::Get().GpuStart());
    commandList->SetComputeRootShaderResourceView(
        GeometryInfoSrv, scene.GetGeometryInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    commandList->SetComputeRootShaderResourceView(
        InstanceInfoSrv, scene.GetInstanceInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    commandList->SetComputeRootShaderResourceView(
        LightDataSrv, scene.GetLightDataBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    commandList->SetComputeRootShaderResourceView(
        EmissiveTrianglesSrv, scene.GetEmissiveTriangleBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    commandList->SetComputeRootShaderResourceView(
        LightPoolSrv, scene.GetLightPoolBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    commandList->SetComputeRootConstantBufferView(PassConstantsCbv, passConstants.GetGpuVirtualAddress());
}
