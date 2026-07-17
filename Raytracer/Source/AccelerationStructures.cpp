#include "pch.h"
#include "AccelerationStructures.h"

#include "BottomLevelASGenerator.h"
#include "DXRHelper.h"
#include "Utils/Utils.h"
#include "InputElements.h"
#include "TopLevelASGenerator.h"
#include "Resources/Buffer.h"
#include "Resources/BufferView.h"

AccelerationStructures::AccelerationStructures()
{
    m_topLevelAsGenerator = std::make_shared<nv_helpers_dx12::TopLevelASGenerator>();
}

AccelerationStructureBuffers AccelerationStructures::CreateBottomLevelAS(
    Microsoft::WRL::ComPtr<ID3D12Device5> device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
    std::vector<BufferView> vertexBuffers,
    std::vector<BufferView> indexBuffers,
    bool isOpaque)
{
    spdlog::debug("Creating bottom level AS");

    // Engine-side build (replaces nv_helpers BottomLevelASGenerator, which
    // hardcodes BUILD_FLAG_NONE): scenes are static and built once, so
    // PREFER_FAST_TRACE buys permanent traversal quality for every traced ray.
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries;
    geometries.reserve(vertexBuffers.size());
    for (size_t i = 0; i < vertexBuffers.size(); i++)
    {
        auto vertexBuffer = vertexBuffers[i].buffer->GetUnderlyingResource();
        auto indexBuffer  = indexBuffers[i].buffer->GetUnderlyingResource();

        spdlog::debug("Adding vertex buffer with name {}: {} vertices and {} indicies.",
            GetName(vertexBuffer.Get()), vertexBuffers[i].count, indexBuffers[i].count);

        D3D12_RAYTRACING_GEOMETRY_DESC desc = {};
        desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        desc.Triangles.VertexBuffer.StartAddress  = vertexBuffer->GetGPUVirtualAddress() + vertexBuffers[i].offsetBytes;
        desc.Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
        desc.Triangles.VertexCount  = static_cast<UINT>(vertexBuffers[i].count);
        desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        desc.Triangles.IndexBuffer  = indexBuffer->GetGPUVirtualAddress() + indexBuffers[i].offsetBytes;
        desc.Triangles.IndexFormat  = DXGI_FORMAT_R32_UINT;
        desc.Triangles.IndexCount   = static_cast<UINT>(indexBuffers[i].count);
        desc.Flags = isOpaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                              : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        geometries.push_back(desc);
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs       = static_cast<UINT>(geometries.size());
    inputs.pGeometryDescs = geometries.data();
    inputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);

    auto alignTo256 = [](uint64_t size) { return (size + 255ull) & ~255ull; };
    const uint64_t scratchSizeInBytes = alignTo256(prebuild.ScratchDataSizeInBytes);
    const uint64_t resultSizeInBytes  = alignTo256(prebuild.ResultDataMaxSizeInBytes);

    AccelerationStructureBuffers buffers;
    buffers.p_scratch = nv_helpers_dx12::CreateBuffer(
        device.Get(), scratchSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
        nv_helpers_dx12::kDefaultHeapProps);
    buffers.p_scratch->SetName(L"Scratch BLAS Buffer");

    buffers.p_result = nv_helpers_dx12::CreateBuffer(
        device.Get(), resultSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nv_helpers_dx12::kDefaultHeapProps);
    buffers.p_result->SetName(L"Result BLAS Buffer");

    spdlog::debug("Generating BLAS (PREFER_FAST_TRACE)");
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs = inputs;
    buildDesc.ScratchAccelerationStructureData = buffers.p_scratch->GetGPUVirtualAddress();
    buildDesc.DestAccelerationStructureData    = buffers.p_result->GetGPUVirtualAddress();
    commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    // The TLAS build (and any trace) must see the finished BLAS.
    D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(buffers.p_result.Get());
    commandList->ResourceBarrier(1, &uavBarrier);

    return buffers;
}

void AccelerationStructures::CreateTopLevelAS(
    Microsoft::WRL::ComPtr<ID3D12Device5> device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList, std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>& instances, bool
    updateOnly)
{
    if (!updateOnly)
    {
        for (size_t i = 0; i < instances.size(); i++)
        {
            auto& instance = instances[i];
            spdlog::debug("Adding instance with name {}", GetName(instance.first.Get()));
            m_topLevelAsGenerator->AddInstance(
                instance.first.Get(), instance.second, static_cast<uint32_t>(i), 0);
        }

        uint64_t scratchSize, resultSize, instanceDescsSize;

        m_topLevelAsGenerator->ComputeASBufferSizes(device.Get(), true, &scratchSize, &resultSize, &instanceDescsSize);

        m_topLevelASBuffers.p_scratch = nv_helpers_dx12::CreateBuffer(
            device.Get(), scratchSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
            nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.p_scratch->SetName(L"Scratch TLAS Buffer");

        m_topLevelASBuffers.p_result = nv_helpers_dx12::CreateBuffer(
            device.Get(), resultSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nv_helpers_dx12::kDefaultHeapProps);
        m_topLevelASBuffers.p_result->SetName(L"Result TLAS Buffer");

        m_topLevelASBuffers.p_instanceDesc = nv_helpers_dx12::CreateBuffer(
            device.Get(), instanceDescsSize,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
            nv_helpers_dx12::kUploadHeapProps);
        m_topLevelASBuffers.p_instanceDesc->SetName(L"Instance Descriptions TLAS Buffer");
    }

    m_topLevelAsGenerator->Generate(commandList.Get(),
        m_topLevelASBuffers.p_scratch.Get(),
        m_topLevelASBuffers.p_result.Get(),
        m_topLevelASBuffers.p_instanceDesc.Get());
}
