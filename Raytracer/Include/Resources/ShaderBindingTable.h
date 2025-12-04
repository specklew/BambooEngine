#pragma once
#include "Buffer.h"

struct SBTEntry
{
    std::wstring entryPoint;
    std::vector<void*> inputData;
};

struct SBTDescriptor
{
    std::vector<SBTEntry> RayGenShaders;
    std::vector<SBTEntry> MissShaders;
    std::vector<SBTEntry> HitShaders;
};

class ShaderBindingTable : public Buffer
{
public:
    ShaderBindingTable(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12StateObjectProperties>& stateObj, SBTDescriptor& sbtDesc);

    size_t GetRayGenEntrySize() const { return m_rayGenEntrySize; }
    size_t GetMissEntrySize() const { return m_missEntrySize; }
    size_t GetHitEntrySize() const { return m_hitEntrySize; }

    size_t GetRayGenSectionSize() const { return m_rayGenEntrySize * m_rayGenShaders.size(); }
    size_t GetMissSectionSize() const { return m_missEntrySize * m_missShaders.size(); }
    size_t GetHitSectionSize() const { return m_hitEntrySize * m_hitShaders.size(); }

private:
    static size_t CalculateEntrySize(std::vector<SBTEntry>& entries);
    
    size_t CalculateSBTSize();
    void CreateSBTBuffer();

    void MapShadersToBuffer();
    size_t CopyShaderData(BYTE* startData, size_t offset, const std::vector<SBTEntry>& shaders, size_t entrySize);

    Microsoft::WRL::ComPtr<ID3D12StateObjectProperties> m_stateObjectProperties;
    
    std::vector<SBTEntry> m_rayGenShaders;
    std::vector<SBTEntry> m_missShaders;
    std::vector<SBTEntry> m_hitShaders;
    
    size_t m_rayGenEntrySize;
    size_t m_missEntrySize;
    size_t m_hitEntrySize;
};
