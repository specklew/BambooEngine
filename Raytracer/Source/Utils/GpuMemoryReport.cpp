#include "pch.h"
#include "Utils/GpuMemoryReport.h"

#include "Resources/Resource.h"

namespace
{
    const char* FormatName(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_UNKNOWN:              return "buffer";
        case DXGI_FORMAT_R32_UINT:             return "R32_UINT";
        case DXGI_FORMAT_R32_SINT:             return "R32_SINT";
        case DXGI_FORMAT_R32_FLOAT:            return "R32_FLOAT";
        case DXGI_FORMAT_R32G32_SINT:          return "RG32_SINT";
        case DXGI_FORMAT_R32G32B32A32_FLOAT:   return "RGBA32_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_SINT:    return "RGBA32_SINT";
        case DXGI_FORMAT_R32G32B32A32_UINT:    return "RGBA32_UINT";
        case DXGI_FORMAT_R8G8B8A8_UNORM:       return "RGBA8_UNORM";
        default:                               return "other";
        }
    }

    std::string Shape(const D3D12_RESOURCE_DESC& desc)
    {
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
            return std::to_string(desc.Width) + " B";
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        {
            // A cube is the case that matters here and reads better as one number.
            if (desc.Width == desc.Height && desc.Height == desc.DepthOrArraySize)
                return std::to_string(desc.Width) + "^3";
            return std::to_string(desc.Width) + "x" + std::to_string(desc.Height) + "x" +
                   std::to_string(desc.DepthOrArraySize);
        }
        // Array slices are why the guide-sample textures cost twice what a glance at the
        // resolution suggests (two samples per pixel), so the shape has to show them.
        if (desc.DepthOrArraySize > 1)
            return std::to_string(desc.Width) + "x" + std::to_string(desc.Height) + "x" +
                   std::to_string(desc.DepthOrArraySize);
        return std::to_string(desc.Width) + "x" + std::to_string(desc.Height);
    }

    std::string Narrow(const std::wstring& wide)
    {
        return std::string(wide.begin(), wide.end());
    }

    std::string Mib(uint64_t bytes)
    {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.2f", double(bytes) / (1024.0 * 1024.0));
        return buffer;
    }
}

void GpuMemoryReport::Add(const char* stage, const std::string& name, ID3D12Resource* resource)
{
    if (resource == nullptr)
        return; // a pass reports what it holds, and before the first resize it holds nothing

    const D3D12_RESOURCE_DESC desc = resource->GetDesc();

    Microsoft::WRL::ComPtr<ID3D12Device> device;
    uint64_t bytes = 0;
    if (SUCCEEDED(resource->GetDevice(IID_PPV_ARGS(&device))))
    {
        // The allocated footprint, not the logical extent: a 3D texture pads, and an
        // answer to P5 that ignored padding would be optimistic by exactly the padding.
        const D3D12_RESOURCE_ALLOCATION_INFO info = device->GetResourceAllocationInfo(0, 1, &desc);
        bytes = info.SizeInBytes;
    }
    if (bytes == 0 || bytes == UINT64_MAX)
        bytes = desc.Width; // buffers, and the fallback if the query refuses

    m_entries.push_back({stage, name, bytes, Shape(desc), FormatName(desc.Format)});
}

void GpuMemoryReport::Add(const char* stage, const Resource* resource)
{
    if (resource == nullptr)
        return;
    Add(stage, Narrow(resource->GetResourceName()), resource->GetUnderlyingResource().Get());
}

void GpuMemoryReport::Add(const char* stage, const std::string& name, const Resource* resource)
{
    if (resource == nullptr)
        return;
    Add(stage, name, resource->GetUnderlyingResource().Get());
}

uint64_t GpuMemoryReport::Total() const
{
    uint64_t total = 0;
    for (const GpuMemoryEntry& entry : m_entries)
        total += entry.bytes;
    return total;
}

std::vector<std::pair<std::string, uint64_t>> GpuMemoryReport::ByStage() const
{
    std::vector<std::pair<std::string, uint64_t>> stages;
    for (const GpuMemoryEntry& entry : m_entries)
    {
        auto it = std::find_if(stages.begin(), stages.end(),
                               [&](const auto& pair) { return pair.first == entry.stage; });
        if (it == stages.end())
            stages.emplace_back(entry.stage, entry.bytes);
        else
            it->second += entry.bytes;
    }
    return stages;
}

std::string GpuMemoryReport::FormatTable() const
{
    std::string out = "GPU memory held by the guiding chain\n";
    out += "  stage                 resource                              shape           format         MiB\n";
    for (const auto& [stage, stageBytes] : ByStage())
    {
        for (const GpuMemoryEntry& entry : m_entries)
        {
            if (entry.stage != stage)
                continue;
            char line[256];
            std::snprintf(line, sizeof(line), "  %-21s %-37s %-15s %-13s %8s\n",
                          entry.stage.c_str(), entry.name.c_str(), entry.shape.c_str(),
                          entry.format.c_str(), Mib(entry.bytes).c_str());
            out += line;
        }
        char subtotal[128];
        std::snprintf(subtotal, sizeof(subtotal), "  %-21s %-67s %8s\n", "", "subtotal",
                      Mib(stageBytes).c_str());
        out += subtotal;
    }
    char total[128];
    std::snprintf(total, sizeof(total), "  %-21s %-67s %8s\n", "", "TOTAL", Mib(Total()).c_str());
    out += total;
    return out;
}
