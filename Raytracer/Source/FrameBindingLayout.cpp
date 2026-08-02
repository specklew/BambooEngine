#include "pch.h"
#include "FrameBindingLayout.h"

#include "GlobalDescriptorHeap.h"
#include "RootSignatureLibrary.h"
#include "SceneResources/Scene.h"
#include "Utils/PassConstants.h"

const std::vector<BindingSlot>& FrameBindingLayout::Slots()
{
    static const std::vector<BindingSlot> slots = {
        // Table first: parameter 0 of every signature that uses this layout.
        kCameraMatrices, kRaytraceOutput, kTlas, kVertices, kIndices, kSkybox, kMaterialTextures,
        // Then the root descriptors, at a fixed prefix of the root parameters.
        kGeometryInfo, kInstanceInfo, kLightData, kEmissiveTriangles, kLightPool, kPassConstants};
    return slots;
}

bool FrameBindingLayout::IsFrameRegister(D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t baseRegister,
                                         uint32_t registerSpace)
{
    for (const BindingSlot& slot : Slots())
    {
        const bool sameType = (type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV && slot.kind == BindingKind::Cbv) ||
                              (type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV && slot.kind == BindingKind::Srv) ||
                              (type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV && slot.kind == BindingKind::Uav);
        if (!sameType || slot.registerSpace != registerSpace)
            continue;

        if (baseRegister >= slot.shaderRegister && baseRegister < slot.shaderRegister + slot.registerCount)
            return true;
    }

    return false;
}

void FrameBindingLayout::Bind(ID3D12GraphicsCommandList* commandList, const RootSignature& rootSignature,
                              Scene& scene, const PassConstants& passConstants)
{
    ID3D12DescriptorHeap* heaps[] = {GlobalDescriptorHeap::Get().GetHeap()};
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    rootSignature.SetTable(commandList, 0, GlobalDescriptorHeap::Get().GpuStart());

    rootSignature.Set(commandList, kGeometryInfo,
                      scene.GetGeometryInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kInstanceInfo,
                      scene.GetInstanceInfoBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kLightData,
                      scene.GetLightDataBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kEmissiveTriangles,
                      scene.GetEmissiveTriangleBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kLightPool,
                      scene.GetLightPoolBuffer()->GetUnderlyingResource()->GetGPUVirtualAddress());
    rootSignature.Set(commandList, kPassConstants, passConstants.GetGpuVirtualAddress());
}
