#include "pch.h"
#include "ScreenshotManager.h"

#include "FrameAccumulationPass.h"
#include "Utils/Utils.h"

// STB_IMAGE_WRITE_IMPLEMENTATION is compiled by Vendor/tinygltf/tiny_gltf.cc.
// Include the header here for the function declarations only.
#include "tinygltf/stb_image_write.h"

#include <ctime>
#include <filesystem>
#include <vector>

void ScreenshotManager::Initialize(
    Microsoft::WRL::ComPtr<ID3D12Device5>              device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    m_device      = device;
    m_commandList = commandList;
}

void ScreenshotManager::Arm(FrameAccumulationPass& accum, float seconds)
{
    if (m_state != State::Idle)
        return;
    
    if (seconds >= FLT_EPSILON)
    {
        accum.Reset();
    }
    
    m_resetCountAtArm = accum.GetResetCount();
    m_targetSeconds   = seconds;
    m_captureDue      = false;
    m_copyRecorded    = false;
    m_wasTargetExceeded = false;
    m_state           = State::Pending;
    spdlog::info("Screenshot armed: accumulating for {:.2f}s", seconds);
}

void ScreenshotManager::Tick(FrameAccumulationPass& accum, double elapsedTime)
{
    if (m_state != State::Pending)
        return;

    if (accum.GetResetCount() != m_resetCountAtArm)
    {
        spdlog::warn("Screenshot cancelled: accumulation was reset (camera moved or window resized)");
        m_state      = State::Idle;
        m_captureDue = false;
        return;
    }

    // Capture on the last frame where accumulatedTime <= T,
    // i.e. the next Update() would push it past T.
    const float t = static_cast<float>(accum.GetAccumulatedTime());
    if (t + static_cast<float>(elapsedTime) > m_targetSeconds)
    {
        spdlog::info("Screenshot capture triggered at {:.3f}s (target {:.2f}s)", t, m_targetSeconds);
        m_captureDue = true;
        m_state      = State::Idle;
        
        if (t > m_targetSeconds)
        {
            m_wasTargetExceeded = true;
            spdlog::warn("Screenshot trigger after target time! The results might have better fidelity than in standard test.");
        }
    }
    
}

void ScreenshotManager::RecordCopy(const Microsoft::WRL::ComPtr<ID3D12Resource>& source)
{
    if (!m_captureDue)
        return;

    const D3D12_RESOURCE_DESC desc = source->GetDesc();
    m_captureWidth  = static_cast<uint32_t>(desc.Width);
    m_captureHeight = static_cast<uint32_t>(desc.Height);

    // Row pitch must be aligned to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256 bytes).
    constexpr UINT kAlign = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    m_rowPitch = (m_captureWidth * 4 + kAlign - 1) & ~(kAlign - 1);
    const uint64_t totalSize = static_cast<uint64_t>(m_rowPitch) * m_captureHeight;

    // (Re)create the readback buffer if the required size grew (e.g. window resized).
    if (totalSize > m_readbackBufferSize)
    {
        m_readbackBuffer.Reset();
        auto heapProps  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(totalSize);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_readbackBuffer)));
        m_readbackBuffer->SetName(L"Screenshot Readback Buffer");
        m_readbackBufferSize = totalSize;
    }

    // source is already in COPY_SOURCE state (PostProcessPass guarantees this).
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
    layout.Offset                 = 0;
    layout.Footprint.Format       = DXGI_FORMAT_R8G8B8A8_UNORM;
    layout.Footprint.Width        = m_captureWidth;
    layout.Footprint.Height       = m_captureHeight;
    layout.Footprint.Depth        = 1;
    layout.Footprint.RowPitch     = m_rowPitch;

    CD3DX12_TEXTURE_COPY_LOCATION srcLoc(source.Get(), 0);
    CD3DX12_TEXTURE_COPY_LOCATION dstLoc(m_readbackBuffer.Get(), layout);
    m_commandList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    m_copyRecorded = true;
}

void ScreenshotManager::FinishCapture()
{
    if (!m_captureDue)
        return;
    m_captureDue   = false;
    if (!m_copyRecorded)
    {
        spdlog::warn("Screenshot skipped: RecordCopy was not called this frame (raytracing inactive?)");
        return;
    }
    m_copyRecorded = false;

    // GPU is done (called after FlushCommandQueue). Map and read back.
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(m_readbackBufferSize) };
    ThrowIfFailed(m_readbackBuffer->Map(0, &readRange, &mappedData));

    // Repack rows: strip D3D12 row-pitch padding into a tight RGBA buffer.
    const uint32_t tightRowBytes = m_captureWidth * 4;
    std::vector<uint8_t> pixels(static_cast<size_t>(tightRowBytes) * m_captureHeight);
    const auto* src = static_cast<const uint8_t*>(mappedData);
    for (uint32_t y = 0; y < m_captureHeight; ++y)
        std::memcpy(pixels.data() + y * tightRowBytes, src + y * m_rowPitch, tightRowBytes);

    D3D12_RANGE writeRange = { 0, 0 };
    m_readbackBuffer->Unmap(0, &writeRange);

    std::filesystem::create_directory("screenshots");
    const std::string filename = MakeFilename();

    if (stbi_write_png(
            filename.c_str(),
            static_cast<int>(m_captureWidth),
            static_cast<int>(m_captureHeight),
            4,
            pixels.data(),
            static_cast<int>(tightRowBytes)))
    {
        spdlog::info("Screenshot saved: {}", filename);
    }
    else
    {
        spdlog::error("Failed to write screenshot: {}", filename);
    }
}

std::string ScreenshotManager::MakeFilename() const
{
    std::time_t t = std::time(nullptr);
    struct tm tm_info = {};
    localtime_s(&tm_info, &t);
    char buf[64];
    std::string format = "screenshots/%Y-%m-%d_%H-%M-%S";
    if (m_wasTargetExceeded) format += "-EXCEEDED";
    format += ".png";
    std::strftime(buf, sizeof(buf), format.c_str(), &tm_info);
    return buf;
}
