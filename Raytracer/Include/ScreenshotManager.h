#pragma once
#include <string>
#include <DirectXMath.h>

class FrameAccumulationPass;

struct ScreenshotMetadata
{
    DirectX::XMFLOAT3 cameraPosition{};
    DirectX::XMFLOAT4 cameraRotation{ 0.0f, 0.0f, 0.0f, 1.0f };
    float             cameraFov = 0.0f;

    std::string modelName;
    std::string placeName;
    std::string techniqueName;

    bool  postProcessEnabled = true;
    float exposure   = 1.0f;
    float contrast   = 1.0f;
    float saturation = 1.0f;
    float lift       = 0.0f;

    uint32_t samplesPerPixel = 0;
    uint32_t bounces         = 0;
    uint32_t frameIndex      = 0;
    float    accumulatedTime = 0.0f;
    uint32_t renderWidth     = 0;
    uint32_t renderHeight    = 0;
};

class ScreenshotManager
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Reset accumulation and start waiting for T seconds; ignored while already pending.
    // metadata is captured at arm time and written to a sidecar JSON when the PNG is saved.
    void Arm(FrameAccumulationPass& accum, float seconds, ScreenshotMetadata metadata);

    // Call each frame BEFORE accumulationPass.Update(elapsedTime)
    void Tick(FrameAccumulationPass& accum, double elapsedTime);

    // Issue CopyTextureRegion into the readback buffer.
    void RecordCopy(const Microsoft::WRL::ComPtr<ID3D12Resource>& source);

    // Map readback buffer and write PNG + sidecar JSON.
    void FinishCapture();

    // Output folder for saved screenshots, relative to the working directory.
    static const char* GetScreenshotsDirectory();

    // Override the destination of the next capture. An empty stem falls back to
    // the auto-generated model-place-timestamp name; an empty dir to the default.
    void SetOutputTarget(const std::string& dir, const std::string& stem);

    bool IsPending()    const { return m_state == State::Pending; }
    bool IsCaptureDue() const { return m_captureDue; }
    bool IsIdle()       const { return m_state == State::Idle && !m_captureDue && !m_copyRecorded; }
    float GetTargetSeconds()    const { return m_targetSeconds; }

private:
    std::string MakeFilenameStem() const;
    void        WriteSidecarJson(const std::string& jsonPath) const;

    enum class State { Idle, Pending };
    State m_state      = State::Idle;
    bool  m_captureDue    = false;
    bool  m_copyRecorded  = false;

    float    m_targetSeconds  = 0.0f;
    uint32_t m_resetCountAtArm = 0;

    ScreenshotMetadata m_pendingMeta;

    std::string m_outDir;   // empty = default screenshots dir
    std::string m_outStem;  // empty = auto-generated name

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_readbackBuffer;
    uint64_t m_readbackBufferSize = 0;

    uint32_t m_captureWidth  = 0;
    uint32_t m_captureHeight = 0;
    UINT     m_rowPitch      = 0;
};
