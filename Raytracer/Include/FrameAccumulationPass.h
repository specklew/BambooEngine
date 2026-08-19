#pragma once
#include "CommandContext.h"

#include "RootSignatureLibrary.h"
#include "ShaderProgram.h"
#include "Resources/RWStructuredBuffer.h"
#include "Resources/Texture.h"

// Mean per-pixel variance of the estimator over an accumulation window, and the
// same divided by the mean squared. Measured without a reference image, so a
// technique with a known error floor can still be compared on how fast its noise
// falls — and so "noisy" and "biased" stop looking alike.
struct VarianceReadout
{
    float mean     = 0.0f;
    float relative = 0.0f;
    bool  valid    = false;
};

class FrameAccumulationPass
{
public:
    void Initialize(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList);

    void Render(Texture& currentFrameOutput);

    void Update(double elapsedTime);
    void Reset();
    void OnResize();

    void SetVarianceEnabled(bool enabled) { m_varianceEnabled = enabled; }
    bool IsVarianceEnabled() const { return m_varianceEnabled; }

    // Reduce this window's per-pixel variance to one scalar pair, then copy it to
    // the readback buffer. Two calls because they are two graph nodes: the
    // transition between them belongs to the graph. Capture-time only — the result
    // cannot be read until the GPU is done with the frame.
    void RecordVarianceReduction();
    void RecordVarianceCopy();
    // Map what the last RecordVarianceCopy produced. Valid only after the frame
    // that recorded it has completed.
    VarianceReadout ReadVarianceResult();

    RWStructuredBuffer<DirectX::XMFLOAT4>& GetVarianceM2() const { return *m_varianceM2; }
    RWStructuredBuffer<DirectX::XMFLOAT2>& GetVarianceResult() const { return *m_varianceResult; }

    uint32_t GetFrameCount()    const { return m_frameCount; }
    double   GetAccumulatedTime() const { return m_accumulatedTime; }
    uint32_t GetResetCount()    const { return m_resetCount; }
    Texture& GetDisplayBuffer() const { return *m_displayBuffer; }

private:
    void CreateResources();
    void CreateRootSignature();
    void CreatePSO();

    Microsoft::WRL::ComPtr<ID3D12Device5>              m_device;
    ActiveCommandList                                  m_commandList;
    std::unique_ptr<Texture>                           m_accumulationBuffer;
    std::unique_ptr<Texture>                           m_displayBuffer;
    RootSignature                                      m_rootSignature;
    ComputeProgram*                                    m_program = nullptr;

    // Allocated whether or not the feature is on: a root UAV has no null form, and
    // the shader still declares the binding even though every write is guarded.
    std::unique_ptr<RWStructuredBuffer<DirectX::XMFLOAT4>> m_varianceM2;
    std::unique_ptr<RWStructuredBuffer<DirectX::XMFLOAT2>> m_varianceResult;
    Microsoft::WRL::ComPtr<ID3D12Resource>                 m_varianceReadback;
    RootSignature                                          m_varianceRootSignature;
    ComputeProgram*                                        m_varianceProgram = nullptr;
    bool                                                   m_varianceEnabled = false;
    bool                                                   m_varianceRecorded = false;

    // Which copy of the pass's global-heap block this frame writes and binds.
    uint32_t m_ringSlot        = 0;
    uint32_t m_frameCount      = 0;
    double   m_accumulatedTime = 0.0;
    uint32_t m_resetCount      = 0;
    bool     m_initialized     = false;
};
