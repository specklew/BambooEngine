#include "pch.h"
#include "Resources/ResourceStateTracker.h"

namespace
{
    std::string NarrowName(const Resource& resource)
    {
        const std::wstring wide = resource.GetResourceName();
        if (wide.empty())
        {
            return "<unnamed>";
        }
        return std::string(wide.begin(), wide.end());
    }
}

ResourceStateTracker& ResourceStateTracker::Get()
{
    static ResourceStateTracker s_directQueueTracker;
    return s_directQueueTracker;
}

void ResourceStateTracker::Register(Resource& resource)
{
    m_resources.insert(&resource);
}

void ResourceStateTracker::Unregister(Resource& resource)
{
    m_resources.erase(&resource);
}

bool ResourceStateTracker::ReportOnce(const std::string& siteKey)
{
    return m_reportedSites.insert(siteKey).second;
}

void ResourceStateTracker::TransitionChecked(ID3D12GraphicsCommandList* commandList, Resource& resource,
                                             D3D12_RESOURCE_STATES expectedBefore, D3D12_RESOURCE_STATES after)
{
    const auto barrier = BuildTransitionChecked(resource, expectedBefore, after);
    commandList->ResourceBarrier(1, &barrier);
}

void ResourceStateTracker::UavBarrierChecked(ID3D12GraphicsCommandList* commandList, Resource& resource)
{
    const auto barrier = BuildUavBarrierChecked(resource);
    commandList->ResourceBarrier(1, &barrier);
}

D3D12_RESOURCE_BARRIER ResourceStateTracker::BuildTransitionChecked(Resource& resource,
                                                                    D3D12_RESOURCE_STATES expectedBefore,
                                                                    D3D12_RESOURCE_STATES after)
{
    ResourceState& tracked = resource.GetTrackedState();

    if (tracked.tracked)
    {
        if (tracked.state == expectedBefore)
        {
            // Model matches the site.
        }
        else if (tracked.state == D3D12_RESOURCE_STATE_COMMON && tracked.CanPromoteFromCommon(expectedBefore))
        {
            // An intervening GPU access promoted the resource out of COMMON; the
            // site's hardcoded before-state is the only evidence of which state.
            spdlog::debug("[StateTracker] {}: promotion assumed COMMON -> {}",
                          NarrowName(resource), ResourceStateConversion::ToString(expectedBefore));
        }
        else
        {
            m_mismatchCount++;
            const std::string siteKey = NarrowName(resource) + "|T|" +
                ResourceStateConversion::ToString(expectedBefore) + "|" +
                ResourceStateConversion::ToString(after);
            if (ReportOnce(siteKey))
            {
                const char* redundantHint = tracked.state == after
                    ? " (redundant: already in after-state)" : "";
                spdlog::error("[StateTracker] MISMATCH {}: tracked {} but site hardcodes {} (-> {}){}",
                              NarrowName(resource),
                              ResourceStateConversion::ToString(tracked.state),
                              ResourceStateConversion::ToString(expectedBefore),
                              ResourceStateConversion::ToString(after),
                              redundantHint);
                if (IsDebuggerPresent())
                {
                    __debugbreak();
                }
            }
        }

        // Both models agree after the barrier regardless of which was right.
        tracked.state = after;
        tracked.promotedReadOnly = false;
    }
    else
    {
        const std::string siteKey = NarrowName(resource) + "|untracked";
        if (ReportOnce(siteKey))
        {
            spdlog::warn("[StateTracker] transition on untracked (non-DEFAULT-heap) resource {}",
                         NarrowName(resource));
        }
    }

    return CD3DX12_RESOURCE_BARRIER::Transition(
        resource.GetUnderlyingResource().Get(), expectedBefore, after);
}

D3D12_RESOURCE_BARRIER ResourceStateTracker::BuildUavBarrierChecked(Resource& resource)
{
    ResourceState& tracked = resource.GetTrackedState();

    if (tracked.tracked)
    {
        if (tracked.state == D3D12_RESOURCE_STATE_COMMON &&
            tracked.CanPromoteFromCommon(D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
        {
            // The UAV write that this barrier flushes promoted the resource.
            tracked.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            tracked.promotedReadOnly = false;
        }
        else if (tracked.state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            const std::string siteKey = NarrowName(resource) + "|U";
            if (ReportOnce(siteKey))
            {
                spdlog::warn("[StateTracker] UAV barrier on {} while tracked state is {}",
                             NarrowName(resource),
                             ResourceStateConversion::ToString(tracked.state));
            }
        }
    }

    return CD3DX12_RESOURCE_BARRIER::UAV(resource.GetUnderlyingResource().Get());
}

void ResourceStateTracker::OnExecuteCommandLists()
{
    for (Resource* resource : m_resources)
    {
        ResourceState& tracked = resource->GetTrackedState();
        if (tracked.tracked && tracked.DecaysAtExecuteCompletion())
        {
            tracked.state = D3D12_RESOURCE_STATE_COMMON;
            tracked.promotedReadOnly = false;
        }
    }
}

namespace ResourceStateConversion
{
    D3D12_BARRIER_LAYOUT ToBarrierLayout(D3D12_RESOURCE_STATES state, bool isBuffer)
    {
        if (isBuffer)
        {
            return D3D12_BARRIER_LAYOUT_UNDEFINED;
        }
        switch (static_cast<uint32_t>(state))
        {
        case D3D12_RESOURCE_STATE_COMMON:                    return D3D12_BARRIER_LAYOUT_COMMON;
        case D3D12_RESOURCE_STATE_RENDER_TARGET:             return D3D12_BARRIER_LAYOUT_RENDER_TARGET;
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:          return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
        case D3D12_RESOURCE_STATE_DEPTH_WRITE:               return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
        case D3D12_RESOURCE_STATE_DEPTH_READ:                return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ;
        case D3D12_RESOURCE_STATE_COPY_DEST:                 return D3D12_BARRIER_LAYOUT_COPY_DEST;
        case D3D12_RESOURCE_STATE_COPY_SOURCE:               return D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        case D3D12_RESOURCE_STATE_RESOLVE_DEST:              return D3D12_BARRIER_LAYOUT_RESOLVE_DEST;
        case D3D12_RESOURCE_STATE_RESOLVE_SOURCE:            return D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE;
        case D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE:       return D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE;
        case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
                                                             return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
        default:
            // Read-state combinations (e.g. SRV | COPY_SOURCE) have no single layout.
            return ResourceState::IsReadOnlyState(state) ? D3D12_BARRIER_LAYOUT_GENERIC_READ
                                                         : D3D12_BARRIER_LAYOUT_COMMON;
        }
    }

    D3D12_BARRIER_ACCESS ToBarrierAccess(D3D12_RESOURCE_STATES state)
    {
        if (state == D3D12_RESOURCE_STATE_COMMON)
        {
            return D3D12_BARRIER_ACCESS_COMMON;
        }

        D3D12_BARRIER_ACCESS access = D3D12_BARRIER_ACCESS_COMMON;
        auto map = [&](D3D12_RESOURCE_STATES legacyBit, D3D12_BARRIER_ACCESS accessBit)
        {
            if (state & legacyBit)
            {
                access |= accessBit;
            }
        };
        map(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_BARRIER_ACCESS_VERTEX_BUFFER | D3D12_BARRIER_ACCESS_CONSTANT_BUFFER);
        map(D3D12_RESOURCE_STATE_INDEX_BUFFER,               D3D12_BARRIER_ACCESS_INDEX_BUFFER);
        map(D3D12_RESOURCE_STATE_RENDER_TARGET,              D3D12_BARRIER_ACCESS_RENDER_TARGET);
        map(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,           D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        map(D3D12_RESOURCE_STATE_DEPTH_WRITE,                D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE);
        map(D3D12_RESOURCE_STATE_DEPTH_READ,                 D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ);
        map(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,  D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
        map(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,      D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
        map(D3D12_RESOURCE_STATE_STREAM_OUT,                 D3D12_BARRIER_ACCESS_STREAM_OUTPUT);
        map(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,          D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT);
        map(D3D12_RESOURCE_STATE_COPY_DEST,                  D3D12_BARRIER_ACCESS_COPY_DEST);
        map(D3D12_RESOURCE_STATE_COPY_SOURCE,                D3D12_BARRIER_ACCESS_COPY_SOURCE);
        map(D3D12_RESOURCE_STATE_RESOLVE_DEST,               D3D12_BARRIER_ACCESS_RESOLVE_DEST);
        map(D3D12_RESOURCE_STATE_RESOLVE_SOURCE,             D3D12_BARRIER_ACCESS_RESOLVE_SOURCE);
        map(D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE);
        map(D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE,        D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE);
        return access;
    }

    std::string ToString(D3D12_RESOURCE_STATES state)
    {
        if (state == D3D12_RESOURCE_STATE_COMMON)
        {
            return "COMMON";
        }

        std::string result;
        auto append = [&](D3D12_RESOURCE_STATES bit, const char* name)
        {
            if (state & bit)
            {
                if (!result.empty())
                {
                    result += "|";
                }
                result += name;
            }
        };
        append(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "VERTEX_AND_CONSTANT_BUFFER");
        append(D3D12_RESOURCE_STATE_INDEX_BUFFER,               "INDEX_BUFFER");
        append(D3D12_RESOURCE_STATE_RENDER_TARGET,              "RENDER_TARGET");
        append(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,           "UNORDERED_ACCESS");
        append(D3D12_RESOURCE_STATE_DEPTH_WRITE,                "DEPTH_WRITE");
        append(D3D12_RESOURCE_STATE_DEPTH_READ,                 "DEPTH_READ");
        append(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,  "NON_PIXEL_SHADER_RESOURCE");
        append(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,      "PIXEL_SHADER_RESOURCE");
        append(D3D12_RESOURCE_STATE_STREAM_OUT,                 "STREAM_OUT");
        append(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,          "INDIRECT_ARGUMENT");
        append(D3D12_RESOURCE_STATE_COPY_DEST,                  "COPY_DEST");
        append(D3D12_RESOURCE_STATE_COPY_SOURCE,                "COPY_SOURCE");
        append(D3D12_RESOURCE_STATE_RESOLVE_DEST,               "RESOLVE_DEST");
        append(D3D12_RESOURCE_STATE_RESOLVE_SOURCE,             "RESOLVE_SOURCE");
        append(D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, "RAYTRACING_ACCELERATION_STRUCTURE");
        append(D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE,        "SHADING_RATE_SOURCE");
        if (result.empty())
        {
            result = "0x" + std::to_string(static_cast<uint32_t>(state));
        }
        return result;
    }
}
