#include "pch.h"
#include "StatesManager.h"

#include "Camera.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace
{
    constexpr const char* kRootDir = "SavedUserData";

    const char* LightTypeToString(LightType t)
    {
        switch (t)
        {
            case LightType::Directional: return "directional";
            case LightType::Spot:        return "spot";
            case LightType::Point:       default: return "point";
        }
    }

    LightType LightTypeFromString(const std::string& s)
    {
        if (s == "directional") return LightType::Directional;
        if (s == "spot")        return LightType::Spot;
        return LightType::Point;
    }

    std::string ReadFile(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file) return {};
        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    bool WriteFile(const std::filesystem::path& path, const std::string& contents)
    {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) return false;
        file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        return file.good();
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

    bool ReadFloat3(const rapidjson::Value& v, DirectX::XMFLOAT3& out)
    {
        if (!v.IsArray() || v.Size() != 3) return false;
        if (!v[0].IsNumber() || !v[1].IsNumber() || !v[2].IsNumber()) return false;
        out.x = v[0].GetFloat(); out.y = v[1].GetFloat(); out.z = v[2].GetFloat();
        return true;
    }

    bool ReadFloat4(const rapidjson::Value& v, DirectX::XMFLOAT4& out)
    {
        if (!v.IsArray() || v.Size() != 4) return false;
        for (rapidjson::SizeType i = 0; i < 4; ++i) if (!v[i].IsNumber()) return false;
        out.x = v[0].GetFloat(); out.y = v[1].GetFloat(); out.z = v[2].GetFloat(); out.w = v[3].GetFloat();
        return true;
    }

    bool FloatsEqual(float a, float b)
    {
        return std::fabs(a - b) <= FLT_EPSILON;
    }

    bool TryStripTrailingNumericSuffix(const std::string& name, std::string& outRoot, int& outNextSuffix)
    {
        const auto underscorePos = name.rfind('_');
        if (underscorePos == std::string::npos || underscorePos + 1 >= name.size())
            return false;

        const std::string tail = name.substr(underscorePos + 1);
        for (char c : tail)
            if (c < '0' || c > '9') return false;

        try
        {
            const int parsed = std::stoi(tail);
            outRoot = name.substr(0, underscorePos);
            outNextSuffix = parsed + 1;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
}

void StatesManager::ParseInto(const std::string& buffer, std::vector<State>& out) const
{
    rapidjson::Document doc;
    doc.Parse(buffer.c_str());

    if (doc.HasParseError() || !doc.IsObject())
    {
        spdlog::warn("states source malformed (offset {}: {}).",
            doc.GetErrorOffset(), rapidjson::GetParseError_En(doc.GetParseError()));
        return;
    }

    for (auto sceneIt = doc.MemberBegin(); sceneIt != doc.MemberEnd(); ++sceneIt)
    {
        if (!sceneIt->value.IsArray()) continue;
        const std::string sceneName = sceneIt->name.GetString();

        for (auto& entry : sceneIt->value.GetArray())
        {
            if (!entry.IsObject()) continue;
            if (!entry.HasMember("name") || !entry["name"].IsString()) continue;

            State state;
            state.sceneName = sceneName;
            state.name      = entry["name"].GetString();

            if (entry.HasMember("position")) ReadFloat3(entry["position"], state.position);
            if (entry.HasMember("rotation")) ReadFloat4(entry["rotation"], state.rotation);
            if (entry.HasMember("fov") && entry["fov"].IsNumber()) state.fov = entry["fov"].GetFloat();

            if (entry.HasMember("lights") && entry["lights"].IsArray())
            {
                state.hasLights = true;
                for (const auto& lo : entry["lights"].GetArray())
                {
                    if (!lo.IsObject()) continue;
                    LightData l{};
                    l.type = lo.HasMember("type") && lo["type"].IsString()
                             ? LightTypeFromString(lo["type"].GetString()) : LightType::Point;
                    if (lo.HasMember("position"))  ReadFloat3(lo["position"],  l.position);
                    if (lo.HasMember("direction")) ReadFloat3(lo["direction"], l.direction);
                    if (lo.HasMember("color"))     ReadFloat3(lo["color"],     l.color);
                    if (lo.HasMember("intensity") && lo["intensity"].IsNumber()) l.intensity = lo["intensity"].GetFloat();
                    if (lo.HasMember("range")     && lo["range"].IsNumber())     l.range     = lo["range"].GetFloat();
                    state.lights.push_back(l);
                }
            }

            out.push_back(std::move(state));
        }
    }
}

void StatesManager::Load()
{
    m_states.clear();
    m_currentScene.clear();
    m_activeState.clear();
    m_haveSnapshot = false;

    const std::filesystem::path path = kSavePath;

    // One-time migration: adopt legacy camera-only places.json when no states.json exists yet.
    const std::filesystem::path legacyPath = std::filesystem::path(kRootDir) / "places.json";
    if (!std::filesystem::exists(path) && std::filesystem::exists(legacyPath))
    {
        ParseInto(ReadFile(legacyPath), m_states);   // legacy entries have no lights -> camera-only
        spdlog::info("Migrated {} legacy places into states.json.", m_states.size());
        Save();
        return;
    }

    if (!std::filesystem::exists(path))
    {
        spdlog::info("states.json not found; starting with empty state set.");
        return;
    }

    ParseInto(ReadFile(path), m_states);
    spdlog::info("Loaded states.json with {} states.", m_states.size());
}

void StatesManager::OnSceneChanged(const std::string& modelName)
{
    m_currentScene = modelName;
    m_activeState.clear();
    m_haveSnapshot = false;
}

void StatesManager::Tick()
{
    if (!m_camera) return;

    if (!m_haveSnapshot)
    {
        SnapshotCamera();
        return;
    }

    if (!m_activeState.empty() && CameraDiffersFromSnapshot())
        m_activeState.clear();

    SnapshotCamera();
}

std::vector<State> StatesManager::GetStatesForCurrentScene() const
{
    std::vector<State> result;
    for (const State& state : m_states)
        if (state.sceneName == m_currentScene)
            result.push_back(state);
    return result;
}

bool StatesManager::AddStateFromScene()
{
    if (!m_camera || m_currentScene.empty())
        return false;

    const size_t sceneSize = GetStatesForCurrentScene().size();

    State state;
    state.sceneName = m_currentScene;
    state.position  = m_camera->GetPosition();
    state.rotation  = m_camera->GetRotation();
    state.fov       = m_camera->GetFovYRadians();
    state.name      = GenerateUniqueName("State_" + std::to_string(sceneSize + 1));

    if (m_getLights)
    {
        state.lights    = m_getLights();
        state.hasLights = true;
    }

    m_states.push_back(std::move(state));
    Save();
    return true;
}

bool StatesManager::RenameState(size_t sceneIndex, const std::string& newName)
{
    if (newName.empty()) return false;

    const size_t globalIndex = ResolveGlobalIndex(sceneIndex);
    if (globalIndex >= m_states.size()) return false;

    State& target = m_states[globalIndex];
    if (target.name == newName) return true;
    if (IsNameTakenInCurrentScene(newName)) return false;

    if (m_activeState == target.name)
        m_activeState = newName;

    target.name = newName;
    Save();
    return true;
}

void StatesManager::DeleteState(size_t sceneIndex)
{
    const size_t globalIndex = ResolveGlobalIndex(sceneIndex);
    if (globalIndex >= m_states.size()) return;

    if (m_activeState == m_states[globalIndex].name)
        m_activeState.clear();

    m_states.erase(m_states.begin() + static_cast<std::ptrdiff_t>(globalIndex));
    Save();
}

void StatesManager::GoToState(size_t sceneIndex)
{
    if (!m_camera) return;

    const size_t globalIndex = ResolveGlobalIndex(sceneIndex);
    if (globalIndex >= m_states.size()) return;

    const State& state = m_states[globalIndex];
    m_camera->SetPosition(state.position);
    m_camera->SetRotation(state.rotation);
    m_camera->SetFovYRadians(state.fov);

    if (state.hasLights && m_setLights)
        m_setLights(state.lights);

    m_activeState = state.name;
    SnapshotCamera();
}

bool StatesManager::GoToStateByName(const std::string& name)
{
    const std::vector<State> sceneStates = GetStatesForCurrentScene();
    for (size_t i = 0; i < sceneStates.size(); ++i)
    {
        if (sceneStates[i].name == name)
        {
            GoToState(i);
            return true;
        }
    }
    return false;
}

bool StatesManager::IsNameTakenInCurrentScene(const std::string& name) const
{
    for (const State& state : m_states)
        if (state.sceneName == m_currentScene && state.name == name)
            return true;
    return false;
}

std::string StatesManager::GenerateUniqueName(const std::string& base) const
{
    if (!IsNameTakenInCurrentScene(base))
        return base;

    std::string root = base;
    int nextSuffix = 1;
    TryStripTrailingNumericSuffix(base, root, nextSuffix);

    while (true)
    {
        std::string candidate = root + "_" + std::to_string(nextSuffix);
        if (!IsNameTakenInCurrentScene(candidate))
            return candidate;
        ++nextSuffix;
    }
}

void StatesManager::Save() const
{
    rapidjson::Document doc(rapidjson::kObjectType);
    auto& alloc = doc.GetAllocator();

    std::unordered_map<std::string, rapidjson::Value> sceneArrays;
    for (const State& state : m_states)
    {
        rapidjson::Value entry(rapidjson::kObjectType);
        rapidjson::Value name; name.SetString(state.name.c_str(), static_cast<rapidjson::SizeType>(state.name.size()), alloc);
        entry.AddMember("name", name, alloc);
        entry.AddMember("position", MakeArray3(state.position, alloc), alloc);
        entry.AddMember("rotation", MakeArray4(state.rotation, alloc), alloc);
        entry.AddMember("fov", state.fov, alloc);

        if (state.hasLights)
        {
            rapidjson::Value lightsArr(rapidjson::kArrayType);
            for (const LightData& l : state.lights)
            {
                rapidjson::Value lo(rapidjson::kObjectType);
                rapidjson::Value typeStr;
                const char* tn = LightTypeToString(l.type);
                typeStr.SetString(tn, static_cast<rapidjson::SizeType>(std::strlen(tn)), alloc);
                lo.AddMember("type", typeStr, alloc);
                lo.AddMember("position",  MakeArray3(l.position,  alloc), alloc);
                lo.AddMember("direction", MakeArray3(l.direction, alloc), alloc);
                lo.AddMember("color",     MakeArray3(l.color,     alloc), alloc);
                lo.AddMember("intensity", l.intensity, alloc);
                lo.AddMember("range",     l.range,     alloc);
                lightsArr.PushBack(lo, alloc);
            }
            entry.AddMember("lights", lightsArr, alloc);
        }

        auto it = sceneArrays.find(state.sceneName);
        if (it == sceneArrays.end())
            it = sceneArrays.emplace(state.sceneName, rapidjson::Value(rapidjson::kArrayType)).first;
        it->second.PushBack(entry, alloc);
    }

    for (auto& [sceneName, array] : sceneArrays)
    {
        rapidjson::Value key;
        key.SetString(sceneName.c_str(), static_cast<rapidjson::SizeType>(sceneName.size()), alloc);
        doc.AddMember(key, array, alloc);
    }

    rapidjson::StringBuffer sb;
    rapidjson::PrettyWriter writer(sb);
    doc.Accept(writer);

    if (!WriteFile(kSavePath, std::string(sb.GetString(), sb.GetSize())))
        spdlog::error("Failed to write states.json");
}

void StatesManager::SnapshotCamera()
{
    if (!m_camera) return;
    m_lastPos      = m_camera->GetPosition();
    m_lastRot      = m_camera->GetRotation();
    m_lastFov      = m_camera->GetFovYRadians();
    m_haveSnapshot = true;
}

bool StatesManager::CameraDiffersFromSnapshot() const
{
    if (!m_camera) return false;

    const DirectX::XMFLOAT3& pos = m_camera->GetPosition();
    const DirectX::XMFLOAT4& rot = m_camera->GetRotation();
    const float              fov = m_camera->GetFovYRadians();

    return !FloatsEqual(pos.x, m_lastPos.x) || !FloatsEqual(pos.y, m_lastPos.y) || !FloatsEqual(pos.z, m_lastPos.z)
        || !FloatsEqual(rot.x, m_lastRot.x) || !FloatsEqual(rot.y, m_lastRot.y) || !FloatsEqual(rot.z, m_lastRot.z) || !FloatsEqual(rot.w, m_lastRot.w)
        || !FloatsEqual(fov,   m_lastFov);
}

size_t StatesManager::ResolveGlobalIndex(size_t sceneIndex) const
{
    size_t seen = 0;
    for (size_t i = 0; i < m_states.size(); ++i)
    {
        if (m_states[i].sceneName != m_currentScene) continue;
        if (seen == sceneIndex) return i;
        ++seen;
    }
    return static_cast<size_t>(-1);
}
