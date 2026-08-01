#include "pch.h"
#include "CommandContext.h"

#include "Resources/Resource.h"
#include "Resources/ResourceStateTracker.h"
#include "Utils/Utils.h"

CommandContext& CommandContext::Get()
{
    static CommandContext s_directQueueContext;
    return s_directQueueContext;
}

void CommandContext::Bind(ID3D12GraphicsCommandList4* commandList)
{
    assert(m_pendingBarriers.empty() && "Barriers queued against the previous command list were never flushed");
    m_commandList = commandList;
}

ID3D12GraphicsCommandList4* CommandContext::GetCommandList()
{
    FlushBarriers();
    return m_commandList;
}

void CommandContext::QueueBarrier(const D3D12_RESOURCE_BARRIER& barrier, ID3D12Resource* resource)
{
    if (std::find(m_pendingBarrierResources.begin(), m_pendingBarrierResources.end(), resource)
        != m_pendingBarrierResources.end())
    {
        FlushBarriers();
    }

    m_pendingBarriers.push_back(barrier);
    m_pendingBarrierResources.push_back(resource);
}

void CommandContext::Transition(Resource& resource, D3D12_RESOURCE_STATES expectedBefore,
                                D3D12_RESOURCE_STATES after)
{
    QueueBarrier(ResourceStateTracker::Get().BuildTransitionChecked(resource, expectedBefore, after),
                 resource.GetUnderlyingResource().Get());
}

void CommandContext::UavBarrier(Resource& resource)
{
    QueueBarrier(ResourceStateTracker::Get().BuildUavBarrierChecked(resource),
                 resource.GetUnderlyingResource().Get());
}

void CommandContext::TransitionRaw(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                   D3D12_RESOURCE_STATES after)
{
    if (!resource)
        return;

    QueueBarrier(CD3DX12_RESOURCE_BARRIER::Transition(resource, before, after), resource);
}

void CommandContext::UavBarrierRaw(ID3D12Resource* resource)
{
    if (!resource)
        return;

    QueueBarrier(CD3DX12_RESOURCE_BARRIER::UAV(resource), resource);
}

void CommandContext::EnqueueBarrier(const D3D12_RESOURCE_BARRIER& barrier, ID3D12Resource* resource)
{
    QueueBarrier(barrier, resource);
}

void CommandContext::FlushBarriers()
{
    if (m_pendingBarriers.empty())
        return;

    assert(m_commandList && "CommandContext used before a command list was bound");
    m_commandList->ResourceBarrier(static_cast<UINT>(m_pendingBarriers.size()), m_pendingBarriers.data());
    m_pendingBarriers.clear();
    m_pendingBarrierResources.clear();
}

void CommandContext::SetComputeProgram(const ComputeProgram& program)
{
    m_commandList->SetComputeRootSignature(program.GetRootSignature());
    m_commandList->SetPipelineState(program.GetPipelineState());
}

void CommandContext::Dispatch(uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ)
{
    FlushBarriers();
    m_commandList->Dispatch(threadGroupsX, threadGroupsY, threadGroupsZ);
}

void CommandContext::DispatchIndirect(ID3D12CommandSignature* commandSignature, ID3D12Resource* argumentBuffer,
                                      uint64_t argumentBufferOffset)
{
    FlushBarriers();
    m_commandList->ExecuteIndirect(commandSignature, 1, argumentBuffer, argumentBufferOffset, nullptr, 0);
}

void CommandContext::DispatchRays(const D3D12_DISPATCH_RAYS_DESC& desc)
{
    FlushBarriers();
    m_commandList->DispatchRays(&desc);
}

void CommandContext::DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndex,
                                          int32_t baseVertex, uint32_t startInstance)
{
    FlushBarriers();
    m_commandList->DrawIndexedInstanced(indexCount, instanceCount, startIndex, baseVertex, startInstance);
}

void CommandContext::CopyResource(ID3D12Resource* destination, ID3D12Resource* source)
{
    FlushBarriers();
    m_commandList->CopyResource(destination, source);
}

void CommandContext::CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION& destination,
                                       const D3D12_TEXTURE_COPY_LOCATION& source)
{
    FlushBarriers();
    m_commandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
}

void CommandContext::Close()
{
    FlushBarriers();
    ThrowIfFailed(m_commandList->Close());
}
