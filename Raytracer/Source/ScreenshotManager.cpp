#include "pch.h"
#include "CommandContext.h"
#include "ScreenshotManager.h"

#include "FrameAccumulationPass.h"
#include "Utils/Utils.h"

// STB_IMAGE_WRITE_IMPLEMENTATION is compiled by Vendor/tinygltf/tiny_gltf.cc.
#include "tinygltf/stb_image_write.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include <chrono>
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

namespace
{
    // Each queued image is width*height*4 bytes (8.3 MB at 1080p). The cap turns a
    // burst of checkpoints into back-pressure on the render thread instead of
    // unbounded memory growth — and back-pressure is just the old synchronous
    // behaviour, so the worst case is what we had before.
    constexpr uint32_t kMaxQueuedWrites = 24;
    constexpr uint32_t kMaxWriterThreads = 4;
}

void ScreenshotManager::Initialize(
    Microsoft::WRL::ComPtr<ID3D12Device5>              device,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList)
{
    m_device      = device;
    m_commandList = commandList;
    StartWriters();
}

void ScreenshotManager::StartWriters()
{
    if (!m_writers.empty())
        return;

    const unsigned hardware = std::thread::hardware_concurrency();
    const unsigned count = std::max(1u, std::min(kMaxWriterThreads, hardware > 2 ? hardware / 2 : 1u));
    for (unsigned i = 0; i < count; ++i)
        m_writers.emplace_back([this]() { WriterLoop(); });
    spdlog::info("Screenshot encoder: {} writer thread(s)", count);
}

void ScreenshotManager::WriterLoop()
{
    for (;;)
    {
        PendingWrite job;
        {
            std::unique_lock<std::mutex> lock(m_writeMutex);
            m_writeReady.wait(lock, [this]() { return m_stopWriters || !m_writeQueue.empty(); });
            if (m_writeQueue.empty())
                return; // stopping, and nothing left to drain
            job = std::move(m_writeQueue.front());
            m_writeQueue.pop_front();
        }
        m_writeDrained.notify_all(); // a slot opened for a blocked producer

        const int rowBytes = static_cast<int>(job.width) * 4;
        if (stbi_write_png(job.pngPath.c_str(), static_cast<int>(job.width), static_cast<int>(job.height),
                           4, job.pixels.data(), rowBytes))
            spdlog::info("Screenshot saved: {}", job.pngPath);
        else
            spdlog::error("Failed to write screenshot: {}", job.pngPath);

        {
            std::lock_guard<std::mutex> lock(m_writeMutex);
            --m_writesInFlight;
        }
        m_writeDrained.notify_all();
    }
}

void ScreenshotManager::WaitForPendingWrites()
{
    std::unique_lock<std::mutex> lock(m_writeMutex);
    m_writeDrained.wait(lock, [this]() { return m_writesInFlight == 0; });
}

void ScreenshotManager::Shutdown()
{
    if (m_writers.empty())
        return;

    WaitForPendingWrites();
    {
        std::lock_guard<std::mutex> lock(m_writeMutex);
        m_stopWriters = true;
    }
    m_writeReady.notify_all();
    for (std::thread& writer : m_writers)
        if (writer.joinable())
            writer.join();
    m_writers.clear();
}

float ScreenshotManager::GetTargetSeconds() const
{
    return m_schedule.budget.kind == CaptureBudget::Kind::Seconds
        ? static_cast<float>(m_schedule.budget.value)
        : 0.0f;
}

void ScreenshotManager::Arm(FrameAccumulationPass& accum, CaptureSchedule schedule, ScreenshotMetadata metadata)
{
    if (m_state != State::Idle)
        return;

    if (schedule.checkpoints.empty())
        schedule.checkpoints.push_back(schedule.budget.value);

    // A frame budget always starts from zero — "one frame" has to mean one frame.
    // A zero-second budget is the interactive "grab what is on screen now" path,
    // which deliberately keeps whatever has already accumulated.
    if (schedule.budget.kind == CaptureBudget::Kind::Frames || schedule.budget.value >= FLT_EPSILON)
        accum.Reset();

    m_resetCountAtArm = accum.GetResetCount();
    m_schedule        = std::move(schedule);
    m_nextCheckpoint  = 0;
    m_frameMsSum      = 0.0;
    m_frameMsCount    = 0;
    m_captureDue      = false;
    m_copyRecorded    = false;
    m_frameInWindow   = false;
    m_pendingMeta     = std::move(metadata);
    m_state           = State::Pending;

    if (m_schedule.budget.kind == CaptureBudget::Kind::Frames)
        spdlog::info("Screenshot armed: {} frame(s), {} checkpoint(s)",
                     static_cast<uint32_t>(m_schedule.budget.value), m_schedule.checkpoints.size());
    else
        spdlog::info("Screenshot armed: accumulating for {:.2f}s, {} checkpoint(s)",
                     m_schedule.budget.value, m_schedule.checkpoints.size());
}

void ScreenshotManager::Tick(FrameAccumulationPass& accum, double elapsedTime, bool accumulating)
{
    if (m_state != State::Pending)
        return;

    if (accumulating && accum.GetResetCount() != m_resetCountAtArm)
    {
        spdlog::warn("Screenshot cancelled: accumulation was reset (camera moved or window resized)");
        m_state      = State::Idle;
        m_captureDue = false;
        return;
    }

    // This frame's duration is not knowable yet - it is measured at the end of the frame, in
    // AccountFrame - so the window's books are closed there and only marked here.
    m_frameInWindow      = true;
    m_windowAccumulating = accumulating;

    // The frame about to be rendered is the one the capture will read, so both budgets are
    // evaluated against the state INCLUDING it: frames already accumulated + 1, and time
    // already accumulated plus a PREDICTION of this frame, for which the previous frame's
    // measured duration is the best available estimate. The prediction only moves where a
    // seconds budget stops; the time this capture reports is the measured sum, finalised in
    // AccountFrame once this frame is actually over.
    const uint32_t framesInImage = accumulating ? accum.GetFrameCount() + 1 : m_frameMsCount + 1;
    const double   timeInImage   = (accumulating ? accum.GetAccumulatedTime()
                                                 : m_frameMsSum / 1000.0) + elapsedTime;
    const double   progress      = m_schedule.budget.kind == CaptureBudget::Kind::Frames
                                 ? static_cast<double>(framesInImage)
                                 : timeInImage;

    if (m_nextCheckpoint >= m_schedule.checkpoints.size() || progress < m_schedule.checkpoints[m_nextCheckpoint])
        return;

    // One image per frame: checkpoints closer together than a frame collapse onto
    // the same one, so skip past every threshold this frame has already passed.
    while (m_nextCheckpoint < m_schedule.checkpoints.size() && progress >= m_schedule.checkpoints[m_nextCheckpoint])
        ++m_nextCheckpoint;

    m_captureDue = true;
    m_pendingMeta.checkpointIndex = static_cast<uint32_t>(m_nextCheckpoint - 1);
    if (m_nextCheckpoint >= m_schedule.checkpoints.size())
        m_state = State::Idle;

    m_pendingMeta.frameIndex      = framesInImage;
    m_pendingMeta.accumulatedTime = static_cast<float>(timeInImage);
    m_pendingMeta.meanFrameMs     = m_frameMsCount > 0 ? static_cast<float>(m_frameMsSum / m_frameMsCount) : 0.0f;

    spdlog::info("Screenshot capture triggered at {} frame(s) / {:.3f}s (predicted; the "
                 "measured total follows)", framesInImage, timeInImage);
}

void ScreenshotManager::AccountFrame(FrameAccumulationPass& accum, double frameSeconds)
{
    // Outside a capture window this is just the running clock the UI shows.
    accum.Update(frameSeconds);
    if (!m_frameInWindow)
        return;
    m_frameInWindow = false;

    m_frameMsSum += frameSeconds * 1000.0;
    ++m_frameMsCount;

    // FinishCapture writes the JSON later in this same frame, so the numbers it reads have to
    // be right now - and now they can be, this frame's duration having just been measured.
    if (m_captureDue)
    {
        const double measured = m_windowAccumulating ? accum.GetAccumulatedTime()
                                                     : m_frameMsSum / 1000.0;
        m_pendingMeta.accumulatedTime = static_cast<float>(measured);
        m_pendingMeta.meanFrameMs     = m_frameMsCount > 0
                                      ? static_cast<float>(m_frameMsSum / m_frameMsCount) : 0.0f;
        spdlog::info("Screenshot window measured: {} frame(s) / {:.3f}s ({:.3f} ms/frame)",
                     m_frameMsCount, measured, m_pendingMeta.meanFrameMs);
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
    CommandContext::Get().GetCommandList()->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    m_copyRecorded = true;
}

const char* ScreenshotManager::GetScreenshotsDirectory()
{
    return kScreenshotsDir;
}

void ScreenshotManager::SetOutputTarget(const std::string& dir, const std::string& stem)
{
    m_outDir  = dir;
    m_outStem = stem;
}

void ScreenshotManager::SetMeasuredVariance(float mean, float relative)
{
    m_pendingMeta.varianceValid    = true;
    m_pendingMeta.varianceMean     = mean;
    m_pendingMeta.varianceRelative = relative;
}

double ScreenshotManager::ConsumeLastCaptureCostSeconds()
{
    const double cost = m_lastCaptureCost;
    m_lastCaptureCost = 0.0;
    return cost;
}

void ScreenshotManager::FinishCapture()
{
    if (!m_captureDue)
        return;
    const auto captureStart = std::chrono::steady_clock::now();
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

    const std::string dir = m_outDir.empty() ? std::string(kScreenshotsDir) : m_outDir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    m_pendingMeta.renderWidth  = m_captureWidth;
    m_pendingMeta.renderHeight = m_captureHeight;

    // A multi-checkpoint schedule writes several images from one accumulation, so
    // the frame count they were taken at is what tells them apart.
    std::string stem = m_outStem.empty() ? MakeFilenameStem() : m_outStem;
    if (m_schedule.checkpoints.size() > 1)
        stem += fmt::format("-f{:06}", m_pendingMeta.frameIndex);

    const std::string pngPath = dir + "/" + stem + ".png";
    const std::string jsonPath = dir + "/" + stem + ".json";

    // The sidecar stays on this thread: it is a few hundred bytes, and writing it
    // here keeps every field it reads (schedule, metadata) single-threaded.
    WriteSidecarJson(jsonPath);

    PendingWrite job;
    job.pixels  = std::move(pixels);
    job.width   = m_captureWidth;
    job.height  = m_captureHeight;
    job.pngPath = pngPath;
    {
        std::unique_lock<std::mutex> lock(m_writeMutex);
        m_writeDrained.wait(lock, [this]() { return m_writeQueue.size() < kMaxQueuedWrites; });
        m_writeQueue.push_back(std::move(job));
        ++m_writesInFlight;
    }
    m_writeReady.notify_one();

    // Now only the map and the row copy — a couple of milliseconds instead of a
    // couple of hundred. Still subtracted from the next frame's delta, because it
    // is still not render time.
    m_lastCaptureCost = std::chrono::duration<double>(std::chrono::steady_clock::now() - captureStart).count();
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
        // What the aggregation script needs to group images and to state the
        // protocol a number came from.
        Value bench(kObjectType);
        bench.AddMember("budgetKind",
            MakeStr(m_schedule.budget.kind == CaptureBudget::Kind::Frames ? "frames" : "seconds", a), a);
        bench.AddMember("budgetValue",   m_schedule.budget.value, a);
        bench.AddMember("checkpoints",     static_cast<uint32_t>(m_schedule.checkpoints.size()), a);
        bench.AddMember("checkpointIndex", m_pendingMeta.checkpointIndex, a);
        bench.AddMember("imageIndex",    m_pendingMeta.imageIndex, a);
        bench.AddMember("imageCount",    m_pendingMeta.imageCount, a);
        bench.AddMember("meanFrameMs",   m_pendingMeta.meanFrameMs, a);
        bench.AddMember("warmupSeconds", m_pendingMeta.warmup.seconds, a);
        // The whole warm-up record, not just its duration: PLAN_BADAWCZY 7.7 wants the
        // state a run settled at in the metadata of every measurement, and the criterion
        // beside it so a later tightening cannot silently reinterpret these files.
        rapidjson::Value warmup(rapidjson::kObjectType);
        warmup.AddMember("seconds",       m_pendingMeta.warmup.seconds,       a);
        warmup.AddMember("meanFrameMs",   m_pendingMeta.warmup.meanFrameMs,   a);
        warmup.AddMember("variation",     m_pendingMeta.warmup.variation,     a);
        warmup.AddMember("drift",         m_pendingMeta.warmup.drift,         a);
        warmup.AddMember("settled",       m_pendingMeta.warmup.settled,       a);
        warmup.AddMember("windowSeconds", m_pendingMeta.warmup.windowSeconds, a);
        warmup.AddMember("threshold",     m_pendingMeta.warmup.threshold,     a);
        warmup.AddMember("minSeconds",    m_pendingMeta.warmup.minSeconds,    a);
        bench.AddMember("warmup", warmup, a);
        bench.AddMember("levers",        MakeStr(m_pendingMeta.activeLevers, a), a);
        bench.AddMember("shaderVariant", MakeStr(m_pendingMeta.shaderVariant, a), a);
        bench.AddMember("settings",      MakeStr(m_pendingMeta.settings, a), a);
        // P5: stage totals, in chain order. Bytes rather than MiB because a sidecar is
        // machine-read and the report is what does the rounding.
        if (m_pendingMeta.memoryTotalBytes > 0)
        {
            rapidjson::Value memory(rapidjson::kObjectType);
            rapidjson::Value stages(rapidjson::kObjectType);
            for (const auto& [stage, bytes] : m_pendingMeta.memoryByStage)
                stages.AddMember(MakeStr(stage, a), bytes, a);
            memory.AddMember("byStage", stages, a);
            memory.AddMember("totalBytes", m_pendingMeta.memoryTotalBytes, a);
            bench.AddMember("memory", memory, a);
        }
        // M8: the process' whole resident footprint on the card, which the inventory
        // above cannot reach, plus the card it was measured on. Written for every arm,
        // path tracing included, because the guided total means nothing without it.
        if (m_pendingMeta.videoMemoryUsedBytes > 0)
            bench.AddMember("videoMemoryBytes", m_pendingMeta.videoMemoryUsedBytes, a);
        if (!m_pendingMeta.adapterName.empty())
            bench.AddMember("adapter", MakeStr(m_pendingMeta.adapterName, a), a);
        if (m_pendingMeta.varianceValid)
        {
            bench.AddMember("varianceMean",     m_pendingMeta.varianceMean,     a);
            bench.AddMember("varianceRelative", m_pendingMeta.varianceRelative, a);
        }
        doc.AddMember("benchmark", bench, a);
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
