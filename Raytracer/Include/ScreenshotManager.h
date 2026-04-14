#pragma once
#include <string>

class FrameAccumulationPass;

class ScreenshotManager
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Reset accumulation and start waiting for T seconds; ignored while already pending
    void Arm(FrameAccumulationPass& accum, float seconds);

    // Call each frame BEFORE accumulationPass.Update(elapsedTime)
    // Checks the pre-update accumulated time to determine whether this is the
    // last frame satisfying accumulatedTime <= T
    void Tick(FrameAccumulationPass& accum, double elapsedTime);

    // Issue CopyTextureRegion into the readback buffer.
    // Call in Render() AFTER post-process and BEFORE command list close.
    // Source must be in D3D12_RESOURCE_STATE_COPY_SOURCE.
    void RecordCopy(const Microsoft::WRL::ComPtr<ID3D12Resource>& source);

    // Map readback buffer and write PNG.
    // Call AFTER FlushCommandQueue so the GPU copy is complete.
    void FinishCapture();

    bool IsPending()    const { return m_state == State::Pending; }
    bool IsCaptureDue() const { return m_captureDue; }
    float GetTargetSeconds()    const { return m_targetSeconds; }

private:
    std::string MakeFilename() const;

    enum class State { Idle, Pending };
    State m_state      = State::Idle;
    bool  m_captureDue    = false;
    bool  m_copyRecorded  = false; // set by RecordCopy, guards FinishCapture

    float    m_targetSeconds  = 0.0f;
    uint32_t m_resetCountAtArm = 0;

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_readbackBuffer;
    uint64_t m_readbackBufferSize = 0;

    uint32_t m_captureWidth  = 0;
    uint32_t m_captureHeight = 0;
    UINT     m_rowPitch      = 0;
};
