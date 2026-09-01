#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Resource;

// What the guiding chain costs in video memory, which is research question P5 and the
// one thing the renderer could not answer at all before ADR 0017 phase 0.
//
// Explicit rather than automatic, and the reason is a fact about this codebase: a
// registry hung off the `Resource` base class would be nearly free but would MISS most
// of the chain, because ADR 0017 deliberately left the voxelization and light-injection
// 3D textures, the V-buffer, the cluster-visibility mask, the superpixel resources and
// the sort keys as raw ComPtr. Those are exactly the resources P5 is about. So each pass
// declares what it holds, and the byproduct is the named inventory of the chain that the
// thesis has to print anyway.
//
// Sizes are the true footprint from GetResourceAllocationInfo, not the logical extent:
// a 3D texture pads, and a report that understated padding would answer P5 optimistically.
struct GpuMemoryEntry
{
    std::string stage;      // which link of the chain it belongs to
    std::string name;
    uint64_t    bytes = 0;
    std::string shape;      // "1920x1080" / "64^3" / "131072 x 32 B"
    std::string format;     // DXGI format name, or "buffer"
};

class GpuMemoryReport
{
public:
    // Raw D3D12 resource: the pass names it, because a raw ComPtr carries no name of
    // its own that survives into a report.
    void Add(const char* stage, const std::string& name, ID3D12Resource* resource);
    // Engine resource: the name comes from the resource itself, so the two cannot drift.
    void Add(const char* stage, const Resource* resource);
    void Add(const char* stage, const std::string& name, const Resource* resource);

    [[nodiscard]] const std::vector<GpuMemoryEntry>& Entries() const { return m_entries; }
    [[nodiscard]] uint64_t Total() const;
    // Stage totals in first-seen order, which is the order of the chain rather than
    // alphabetical — the table reads as the pipeline it describes.
    [[nodiscard]] std::vector<std::pair<std::string, uint64_t>> ByStage() const;

    // The full inventory, one line per resource, plus stage subtotals and the total.
    [[nodiscard]] std::string FormatTable() const;

    [[nodiscard]] bool Empty() const { return m_entries.empty(); }

private:
    std::vector<GpuMemoryEntry> m_entries;
};

// Stage names, in chain order. Constants rather than string literals at the call sites
// so a typo cannot silently split one stage into two rows of the report.
namespace GpuMemoryStage
{
    inline constexpr const char* Voxelization   = "voxelization";
    inline constexpr const char* VBuffer        = "v-buffer";
    inline constexpr const char* Injection      = "light injection";
    inline constexpr const char* GuidingBuild   = "guiding build";
    inline constexpr const char* Fingerprint    = "fingerprint";
    inline constexpr const char* Superpixel     = "superpixels";
    inline constexpr const char* Cluster        = "clustering";
    inline constexpr const char* ClusterVis     = "cluster visibility";
    inline constexpr const char* LightTree      = "light tree";
    inline constexpr const char* Integrator     = "guided integrator";
    // Not the guiding chain, and reported separately so the P5 figure is not inflated
    // by things path tracing pays for too.
    inline constexpr const char* Scene          = "scene (shared)";
    inline constexpr const char* Output         = "output (shared)";
}
