#pragma once

#include <cstring>
#include <d3d12.h>

// RAII GPU event marker for PIX captures: brackets the enclosed command-list
// work in a named region so the PIX Timing view reports per-pass GPU cost by
// name instead of anonymous dispatch indices. Uses the raw
// ID3D12GraphicsCommandList::BeginEvent ANSI form (metadata = 1, the
// PIX_EVENT_ANSI_VERSION encoding), so name-only regions need no
// WinPixEventRuntime dependency. Negligible cost when no tool is attached.
struct ScopedGpuMarker
{
    ScopedGpuMarker(ID3D12GraphicsCommandList* commandList, const char* name)
        : m_commandList(commandList)
    {
        if (m_commandList)
            m_commandList->BeginEvent(1, name, static_cast<UINT>(std::strlen(name) + 1));
    }

    ~ScopedGpuMarker()
    {
        if (m_commandList)
            m_commandList->EndEvent();
    }

    ScopedGpuMarker(const ScopedGpuMarker&) = delete;
    ScopedGpuMarker& operator=(const ScopedGpuMarker&) = delete;

private:
    ID3D12GraphicsCommandList* m_commandList;
};
