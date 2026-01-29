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
    std::vector<BufferView> indexBuffers)
{
    spdlog::debug("Creating bottom level AS");
    
    nv_helpers_dx12::BottomLevelASGenerator bottomLevelGenerator;
    
    for (int i = 0; i < vertexBuffers.size(); i++)
    {
        auto vertexBuffer = vertexBuffers[i].buffer->GetUnderlyingResource();
        auto vertexCount = vertexBuffers[i].count;
        auto vertexOffset = vertexBuffers[i].offsetBytes;
        
        auto indexBuffer = indexBuffers[i].buffer->GetUnderlyingResource();
        auto indexCount = indexBuffers[i].count;
        auto indexOffset = indexBuffers[i].offsetBytes;

        spdlog::debug("Adding vertex buffer with name {}: {} vertices and {} indicies.", GetName(vertexBuffer.Get()), vertexCount, indexCount);
        bottomLevelGenerator.AddVertexBuffer(vertexBuffer.Get(), vertexOffset, vertexCount, sizeof(Vertex),
            indexBuffer.Get(), indexOffset, indexCount, nullptr, 0);
    }

    uint64_t scratchSizeInBytes = 0;
    uint64_t resultSizeInBytes = 0;
    
    bottomLevelGenerator.ComputeASBufferSizes(device.Get(), false, &scratchSizeInBytes, &resultSizeInBytes);

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

    spdlog::debug("Generating BLAS");
    bottomLevelGenerator.Generate(commandList.Get(), buffers.p_scratch.Get(), buffers.p_result.Get(), false);

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
                instance.first.Get(), instance.second, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
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
