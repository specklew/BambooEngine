#include "pch.h"
#include "ShaderBindingTable.h"

#include "Helpers.h"

static constexpr size_t EntryIDSize = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;

ShaderBindingTable::ShaderBindingTable(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12StateObjectProperties>& stateObj, SBTDescriptor& sbtDesc)
    : Buffer(device, nullptr),
    m_stateObjectProperties(stateObj),
    m_rayGenShaders(std::move(sbtDesc.RayGenShaders)),
    m_missShaders(std::move(sbtDesc.MissShaders)),
    m_hitShaders(std::move(sbtDesc.HitShaders))
{
    if (m_rayGenShaders.empty() || m_missShaders.empty() || m_hitShaders.empty())
    {
        spdlog::warn("One or more of the SBT shader groups is empty. RayGen shaders: {}, Miss shaders: {}, Hit shaders: {}",
            m_rayGenShaders.size(), m_missShaders.size(), m_hitShaders.size());        
    }
    
    m_rayGenEntrySize = CalculateEntrySize(m_rayGenShaders);
    m_missEntrySize = CalculateEntrySize(m_missShaders);
    m_hitEntrySize = CalculateEntrySize(m_hitShaders);

    CreateSBTBuffer();
    MapShadersToBuffer();

    QueryFeatureSupport();
}

size_t ShaderBindingTable::CalculateEntrySize(std::vector<SBTEntry>& entries)
{
    size_t max_parameters = 0;
    for (const auto& entry : entries)
    {
        max_parameters = std::max(max_parameters, entry.inputData.size());
    }
    
    size_t entry_size = EntryIDSize + max_parameters * sizeof(void*);

    entry_size = Align(entry_size, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);

    return entry_size;
}

size_t ShaderBindingTable::CalculateSBTSize()
{
    size_t sbt_size = 0;

    sbt_size += m_rayGenEntrySize * m_rayGenShaders.size();
    sbt_size += m_missEntrySize * m_missShaders.size();
    sbt_size += m_hitEntrySize * m_hitShaders.size();

    sbt_size = Align(sbt_size, 256); // Align to match the 256 bytes requirement for buffers
    
    return sbt_size;
}

void ShaderBindingTable::CreateSBTBuffer()
{
    auto sbtSize = CalculateSBTSize();

    D3D12_RESOURCE_DESC bufDesc;
    bufDesc.Alignment = 0;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    bufDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufDesc.Height = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.SampleDesc.Quality = 0;
    bufDesc.Width = sbtSize;

    ThrowIfFailed(m_device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &bufDesc,
                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_resource)));

    SetResourceName(L"SBT Buffer");
}

void ShaderBindingTable::MapShadersToBuffer()
{
    BYTE* mappedData;
    ThrowIfFailed(m_resource->Map(0, nullptr, reinterpret_cast<void**>(&mappedData)));
    size_t offset = 0;

    offset = CopyShaderData(mappedData, offset, m_rayGenShaders, m_rayGenEntrySize);
    offset = CopyShaderData(mappedData, offset, m_missShaders, m_missEntrySize);
    offset = CopyShaderData(mappedData, offset, m_hitShaders, m_hitEntrySize);

    m_resource->Unmap(0, nullptr);
}

size_t ShaderBindingTable::CopyShaderData(BYTE* startData, size_t offset, const std::vector<SBTEntry>& shaders, size_t entrySize)
{
    BYTE* pData = startData + offset;
    
    for (const auto& entry : shaders)
    {
        void* shader_id = m_stateObjectProperties->GetShaderIdentifier(entry.entryPoint.c_str());
        
        if (!shader_id) spdlog::error("Failed to get shader identifier!");

        memcpy(pData, shader_id, EntryIDSize);
        memcpy(pData + EntryIDSize, entry.inputData.data(), entry.inputData.size() * sizeof(void*));

        offset += entrySize;
    }

    return offset;
}
