#include "pch.h"
#include "ScreenshotManager.h"

#include "FrameAccumulationPass.h"
#include "Utils/Utils.h"

// STB_IMAGE_WRITE_IMPLEMENTATION is compiled by Vendor/tinygltf/tiny_gltf.cc.
#include "tinygltf/stb_image_write.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{
    constexpr const char* kScreenshotsDir = "SavedUserData/Screenshots";

    std::string Sanitize(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s)
        {
            if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
                c == '"' || c == '<' || c == '>' || c == '|')
                out.push_back('_');
            else
                out.push_back(c);
        }
        return out;
    }

    std::string TimestampForFilename()
    {
        std::time_t t = std::time(nullptr);
        struct tm tm_info = {};
        localtime_s(&tm_info, &t);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm_info);
        return buf;
    }

    std::string TimestampISO()
    {
        std::time_t t = std::time(nullptr);
        struct tm tm_info = {};
        localtime_s(&tm_info, &t);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_info);
        return buf;
    }

    rapidjson::Value MakeArray3(const DirectX::XMFLOAT3& v, rapidjson::Document::AllocatorType& alloc)
    {
        rapidjson::Value a(rapidjson::kArrayType);
        a.PushBack(v.x, alloc); a.PushBack(v.y, alloc); a.PushBack(v.z, alloc);
        return a;
    }

    rapidjson::Value MakeArray4(const DirectX::XMFLOAT4& v, rapidjson::Document::AllocatorType& alloc)
    {
        rapidjson::Value a(rapidjson::kArrayType);
        a.PushBack(v.x, alloc); a.PushBack(v.y, alloc); a.PushBack(v.z, alloc); a.PushBack(v.w, alloc);
        return a;
    }

    rapidjson::Value MakeStr(const std::string& s, rapidjson::Document::AllocatorType& alloc)
    {
        rapidjson::Value v;
        v.SetString(s.c_str(), static_cast<rapidjson::SizeType>(s.size()), alloc);
        return v;
    }
}

void ScreenshotManager::Initialize(
    Microsoft::WRL::ComPtr<ID3D12Device5>              device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    m_device      = device;
    m_commandList = commandList;
}

void ScreenshotManager::Arm(FrameAccumulationPass& accum, float seconds, ScreenshotMetadata metadata)
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
    m_pendingMeta     = std::move(metadata);
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

    const float t = static_cast<float>(accum.GetAccumulatedTime());
    if (t + static_cast<float>(elapsedTime) > m_targetSeconds)
    {
        spdlog::info("Screenshot capture triggered at {:.3f}s (target {:.2f}s)", t, m_targetSeconds);
        m_captureDue = true;
        m_state      = State::Idle;

        m_pendingMeta.frameIndex      = accum.GetFrameCount();
        m_pendingMeta.accumulatedTime = t;

        if (t > m_targetSeconds)
            spdlog::warn("Screenshot trigger after target time! The results might have better fidelity than in standard test.");
    }
}

void ScreenshotManager::RecordCopy(const Microsoft::WRL::ComPtr<ID3D12Resource>& source)
{
    if (!m_captureDue)
        return;

    const D3D12_RESOURCE_DESC desc = source->GetDesc();
    m_captureWidth  = static_cast<uint32_t>(desc.Width);
    m_captureHeight = static_cast<uint32_t>(desc.Height);

    constexpr UINT kAlign = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    m_rowPitch = (m_captureWidth * 4 + kAlign - 1) & ~(kAlign - 1);
    const uint64_t totalSize = static_cast<uint64_t>(m_rowPitch) * m_captureHeight;

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

const char* ScreenshotManager::GetScreenshotsDirectory()
{
    return kScreenshotsDir;
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

    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(m_readbackBufferSize) };
    ThrowIfFailed(m_readbackBuffer->Map(0, &readRange, &mappedData));

    const uint32_t tightRowBytes = m_captureWidth * 4;
    std::vector<uint8_t> pixels(static_cast<size_t>(tightRowBytes) * m_captureHeight);
    const auto* src = static_cast<const uint8_t*>(mappedData);
    for (uint32_t y = 0; y < m_captureHeight; ++y)
        std::memcpy(pixels.data() + y * tightRowBytes, src + y * m_rowPitch, tightRowBytes);

    D3D12_RANGE writeRange = { 0, 0 };
    m_readbackBuffer->Unmap(0, &writeRange);

    std::error_code ec;
    std::filesystem::create_directories(kScreenshotsDir, ec);

    m_pendingMeta.renderWidth  = m_captureWidth;
    m_pendingMeta.renderHeight = m_captureHeight;

    const std::string stem    = MakeFilenameStem();
    const std::string pngPath = std::string(kScreenshotsDir) + "/" + stem + ".png";
    const std::string jsonPath = std::string(kScreenshotsDir) + "/" + stem + ".json";

    if (stbi_write_png(
            pngPath.c_str(),
            static_cast<int>(m_captureWidth),
            static_cast<int>(m_captureHeight),
            4,
            pixels.data(),
            static_cast<int>(tightRowBytes)))
    {
        spdlog::info("Screenshot saved: {}", pngPath);
        WriteSidecarJson(jsonPath);
    }
    else
    {
        spdlog::error("Failed to write screenshot: {}", pngPath);
    }
}

std::string ScreenshotManager::MakeFilenameStem() const
{
    const std::string model = m_pendingMeta.modelName.empty() ? std::string("unknown") : Sanitize(m_pendingMeta.modelName);
    const std::string place = Sanitize(m_pendingMeta.placeName);
    const std::string ts    = TimestampForFilename();

    std::string stem = model;
    if (!place.empty()) { stem += "-"; stem += place; }
    stem += "-"; stem += ts;
    return stem;
}

void ScreenshotManager::WriteSidecarJson(const std::string& jsonPath) const
{
    using namespace rapidjson;
    Document doc(kObjectType);
    auto& a = doc.GetAllocator();

    {
        Value cam(kObjectType);
        cam.AddMember("position", MakeArray3(m_pendingMeta.cameraPosition, a), a);
        cam.AddMember("rotation", MakeArray4(m_pendingMeta.cameraRotation, a), a);
        cam.AddMember("fov",      m_pendingMeta.cameraFov, a);
        doc.AddMember("camera", cam, a);
    }
    {
        Value sc(kObjectType);
        sc.AddMember("model", MakeStr(m_pendingMeta.modelName, a), a);
        sc.AddMember("place", MakeStr(m_pendingMeta.placeName, a), a);
        doc.AddMember("scene", sc, a);
    }
    doc.AddMember("technique", MakeStr(m_pendingMeta.techniqueName, a), a);
    {
        Value pp(kObjectType);
        pp.AddMember("enabled",    m_pendingMeta.postProcessEnabled, a);
        pp.AddMember("exposure",   m_pendingMeta.exposure,   a);
        pp.AddMember("contrast",   m_pendingMeta.contrast,   a);
        pp.AddMember("saturation", m_pendingMeta.saturation, a);
        pp.AddMember("lift",       m_pendingMeta.lift,       a);
        doc.AddMember("postProcess", pp, a);
    }
    {
        Value rt(kObjectType);
        rt.AddMember("spp",             m_pendingMeta.samplesPerPixel, a);
        rt.AddMember("bounces",         m_pendingMeta.bounces,         a);
        rt.AddMember("frameIndex",      m_pendingMeta.frameIndex,      a);
        rt.AddMember("accumulatedTime", m_pendingMeta.accumulatedTime, a);
        doc.AddMember("raytracing", rt, a);
    }
    {
        Value rd(kObjectType);
        rd.AddMember("width",  m_pendingMeta.renderWidth,  a);
        rd.AddMember("height", m_pendingMeta.renderHeight, a);
        doc.AddMember("render", rd, a);
    }
    {
        Value cap(kObjectType);
        cap.AddMember("timestamp", MakeStr(TimestampISO(), a), a);
        doc.AddMember("capture", cap, a);
    }

    StringBuffer sb;
    PrettyWriter<StringBuffer> writer(sb);
    doc.Accept(writer);

    std::ofstream f(jsonPath, std::ios::binary | std::ios::trunc);
    if (!f)
    {
        spdlog::error("Failed to open sidecar metadata for write: {}", jsonPath);
        return;
    }
    f.write(sb.GetString(), static_cast<std::streamsize>(sb.GetSize()));
}
