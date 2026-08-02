#pragma once
#include <set>
#include <unordered_set>

#include "Resources/Resource.h"

// ADR 0017 phase-0 assert harness + resource state tracker. Throwaway by design:
// once phase-3 barrier synthesis emits from tracked state alone, the expectedBefore
// checking and per-site reporting are deleted in one mechanical pass.
//
// One instance per queue; the engine runs a single direct queue today, so Get()
// returns the direct-queue tracker.
class ResourceStateTracker
{
public:
    static ResourceStateTracker& Get();

    void Register(Resource& resource);
    void Unregister(Resource& resource);

    // Builds exactly the barrier the call site always emitted (expectedBefore → after)
    // and checks expectedBefore against tracked state, promotion-aware. A mismatch is
    // either a tracker bug or a wrong hardcoded state — both findings, reported once
    // per (resource, before, after) site. CommandContext (L3) batches the returned
    // barriers and submits them together ahead of the work that needs them.
    D3D12_RESOURCE_BARRIER BuildTransitionChecked(Resource& resource, D3D12_RESOURCE_STATES expectedBefore,
                                                  D3D12_RESOURCE_STATES after);

    // Builds the UAV barrier; warns when the tracked state is not (promotable to)
    // UNORDERED_ACCESS.
    D3D12_RESOURCE_BARRIER BuildUavBarrierChecked(Resource& resource);

    // Applies implicit decay to COMMON. Call after every ExecuteCommandLists on the
    // queue this tracker models; the GPU decays at execution completion, but on the
    // CPU-side recording timeline the ECL call is exactly where later command lists
    // start seeing the decayed state.
    void OnExecuteCommandLists();

    [[nodiscard]] uint32_t GetMismatchCount() const { return m_mismatchCount; }

private:
    std::unordered_set<Resource*> m_resources;
    std::set<std::string> m_reportedSites;
    uint32_t m_mismatchCount = 0;

    bool ReportOnce(const std::string& siteKey);
};

// Legacy state → enhanced-barrier equivalents (per the D3D12 enhanced-barriers
// mapping table). Sync scopes are intentionally absent: phase-3 synthesis derives
// them from producer/consumer passes, not from stored state.
namespace ResourceStateConversion
{
    D3D12_BARRIER_LAYOUT ToBarrierLayout(D3D12_RESOURCE_STATES state, bool isBuffer);
    D3D12_BARRIER_ACCESS ToBarrierAccess(D3D12_RESOURCE_STATES state);

    std::string ToString(D3D12_RESOURCE_STATES state);
}
