#pragma once
#include "CommandContext.h"
#include <string>
#include <vector>
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

    // Benchmark provenance: which image of a multi-image experiment this is, and
    // what the measured frame cost was over its accumulation window. A capture
    // whose statistics are quoted without these is not reproducible.
    uint32_t imageIndex      = 0; // 0-based, 0 for a single-image run
    uint32_t imageCount      = 1;
    // Which checkpoint of the schedule this image is. The frame count cannot
    // stand in for it: under a time budget every image crosses a checkpoint at a
    // different frame, so the ordinal is what lines curves up across images.
    uint32_t checkpointIndex = 0;
    float    meanFrameMs     = 0.0f;
    float    warmupSeconds   = 0.0f; // what the run actually spent warming up

    // Estimator variance over this image's accumulation window, measured without
    // a reference (renderer.accumulation.variance). Absent when the feature is off.
    bool  varianceValid    = false;
    float varianceMean     = 0.0f;
    float varianceRelative = 0.0f;
};

// A capture's stopping condition. Seconds is the equal-time axis, frames the
// equal-sample-count one; a run states exactly one of them.
struct CaptureBudget
{
    enum class Kind { Seconds, Frames };

    Kind   kind  = Kind::Seconds;
    double value = 0.0; // seconds, or frames

    static CaptureBudget Seconds(double s) { return { Kind::Seconds, s }; }
    static CaptureBudget Frames(uint32_t n) { return { Kind::Frames, static_cast<double>(n) }; }
};

// One accumulation run and the progress values at which it writes an image.
// Checkpoints are in the budget's own unit and must be ascending; the last one
// is the budget itself. A single-entry schedule is the classic capture-at-the-end.
// Several checkpoints turn one run into a convergence curve — the same
// accumulation sampled repeatedly, which is a different measurement from several
// independent images and is meant to be used alongside them, not instead.
struct CaptureSchedule
{
    CaptureBudget       budget;
    std::vector<double> checkpoints;

    static CaptureSchedule AtEnd(CaptureBudget budget) { return { budget, { budget.value } }; }
};

class ScreenshotManager
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5>              device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    // Reset accumulation and run the schedule; ignored while already pending.
    // metadata is captured at arm time and written to a sidecar JSON per image.
    void Arm(FrameAccumulationPass& accum, CaptureSchedule schedule, ScreenshotMetadata metadata);

    // Call each frame BEFORE accumulationPass.Update(elapsedTime)
    void Tick(FrameAccumulationPass& accum, double elapsedTime);

    // Issue CopyTextureRegion into the readback buffer.
    void RecordCopy(const Microsoft::WRL::ComPtr<ID3D12Resource>& source);

    // Map readback buffer and write PNG + sidecar JSON.
    void FinishCapture();

    // Measured just before FinishCapture, once the frame that recorded it is done.
    void SetMeasuredVariance(float mean, float relative);

    // Seconds the last FinishCapture spent mapping and encoding, cleared by the
    // read. The clock hands that cost to the NEXT frame, where it would otherwise
    // be counted as render time — and a PNG encode is ~90 ms against a ~5 ms
    // frame, so a checkpointed run would report a fraction of the frames it
    // really rendered.
    double ConsumeLastCaptureCostSeconds();

    // Output folder for saved screenshots, relative to the working directory.
    static const char* GetScreenshotsDirectory();

    // Override the destination of the next capture. An empty stem falls back to
    // the auto-generated model-place-timestamp name; an empty dir to the default.
    void SetOutputTarget(const std::string& dir, const std::string& stem);

    bool IsPending()    const { return m_state == State::Pending; }
    bool IsCaptureDue() const { return m_captureDue; }
    bool IsIdle()       const { return m_state == State::Idle && !m_captureDue && !m_copyRecorded; }
    float GetTargetSeconds() const;

private:
    std::string MakeFilenameStem() const;
    void        WriteSidecarJson(const std::string& jsonPath) const;

    enum class State { Idle, Pending };
    State m_state      = State::Idle;
    bool  m_captureDue    = false;
    bool  m_copyRecorded  = false;

    CaptureSchedule m_schedule;
    size_t          m_nextCheckpoint = 0;
    uint32_t        m_resetCountAtArm = 0;
    double          m_frameMsSum      = 0.0;
    uint32_t        m_frameMsCount    = 0;
    double          m_lastCaptureCost = 0.0;

    ScreenshotMetadata m_pendingMeta;

    std::string m_outDir;   // empty = default screenshots dir
    std::string m_outStem;  // empty = auto-generated name

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;
    Microsoft::WRL::ComPtr<ID3D12Resource>             m_readbackBuffer;
    uint64_t m_readbackBufferSize = 0;

    uint32_t m_captureWidth  = 0;
    uint32_t m_captureHeight = 0;
    UINT     m_rowPitch      = 0;
};
