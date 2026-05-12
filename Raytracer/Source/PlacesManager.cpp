#include "pch.h"
#include "PlacesManager.h"

#include "Camera.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace
{
    constexpr const char* kRootDir = "SavedUserData";

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

void PlacesManager::Load()
{
    m_places.clear();
    m_currentScene.clear();
    m_activePlace.clear();
    m_haveSnapshot = false;

    const std::filesystem::path path = kSavePath;
    if (!std::filesystem::exists(path))
    {
        spdlog::info("places.json not found; starting with empty bookmark set.");
        return;
    }

    const std::string buffer = ReadFile(path);
    rapidjson::Document doc;
    doc.Parse(buffer.c_str());

    if (doc.HasParseError() || !doc.IsObject())
    {
        spdlog::warn("places.json malformed (offset {}: {}). Backing up to places.json.bak and starting fresh.",
            doc.GetErrorOffset(), rapidjson::GetParseError_En(doc.GetParseError()));
        std::error_code ec;
        std::filesystem::rename(path, std::filesystem::path(kRootDir) / "places.json.bak", ec);
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

            Place place;
            place.sceneName = sceneName;
            place.name      = entry["name"].GetString();

            if (entry.HasMember("position")) ReadFloat3(entry["position"], place.position);
            if (entry.HasMember("rotation")) ReadFloat4(entry["rotation"], place.rotation);
            if (entry.HasMember("fov") && entry["fov"].IsNumber()) place.fov = entry["fov"].GetFloat();

            m_places.push_back(std::move(place));
        }
    }

    spdlog::info("Loaded places.json with {} places.", m_places.size());
}

void PlacesManager::OnSceneChanged(const std::string& modelName)
{
    m_currentScene = modelName;
    m_activePlace.clear();
    m_haveSnapshot = false;
}

void PlacesManager::Tick()
{
    if (!m_camera) return;

    if (!m_haveSnapshot)
    {
        SnapshotCamera();
        return;
    }

    if (!m_activePlace.empty() && CameraDiffersFromSnapshot())
        m_activePlace.clear();

    SnapshotCamera();
}

std::vector<Place> PlacesManager::GetPlacesForCurrentScene() const
{
    std::vector<Place> result;
    for (const Place& place : m_places)
        if (place.sceneName == m_currentScene)
            result.push_back(place);
    return result;
}

bool PlacesManager::AddPlaceFromCamera()
{
    if (!m_camera || m_currentScene.empty())
        return false;

    const size_t sceneSize = GetPlacesForCurrentScene().size();

    Place place;
    place.sceneName = m_currentScene;
    place.position  = m_camera->GetPosition();
    place.rotation  = m_camera->GetRotation();
    place.fov       = m_camera->GetFovYRadians();
    place.name      = GenerateUniqueName("Place_" + std::to_string(sceneSize + 1));

    m_places.push_back(std::move(place));
    Save();
    return true;
}

bool PlacesManager::RenamePlace(size_t sceneIndex, const std::string& newName)
{
    if (newName.empty()) return false;

    const size_t globalIndex = ResolveGlobalIndex(sceneIndex);
    if (globalIndex >= m_places.size()) return false;

    Place& target = m_places[globalIndex];
    if (target.name == newName) return true;
    if (IsNameTakenInCurrentScene(newName)) return false;

    if (m_activePlace == target.name)
        m_activePlace = newName;

    target.name = newName;
    Save();
    return true;
}

void PlacesManager::DeletePlace(size_t sceneIndex)
{
    const size_t globalIndex = ResolveGlobalIndex(sceneIndex);
    if (globalIndex >= m_places.size()) return;

    if (m_activePlace == m_places[globalIndex].name)
        m_activePlace.clear();

    m_places.erase(m_places.begin() + static_cast<std::ptrdiff_t>(globalIndex));
    Save();
}

void PlacesManager::GoToPlace(size_t sceneIndex)
{
    if (!m_camera) return;

    const size_t globalIndex = ResolveGlobalIndex(sceneIndex);
    if (globalIndex >= m_places.size()) return;

    const Place& place = m_places[globalIndex];
    m_camera->SetPosition(place.position);
    m_camera->SetRotation(place.rotation);
    m_camera->SetFovYRadians(place.fov);

    m_activePlace = place.name;
    SnapshotCamera();
}

bool PlacesManager::IsNameTakenInCurrentScene(const std::string& name) const
{
    for (const Place& place : m_places)
        if (place.sceneName == m_currentScene && place.name == name)
            return true;
    return false;
}

std::string PlacesManager::GenerateUniqueName(const std::string& base) const
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

void PlacesManager::Save() const
{
    rapidjson::Document doc(rapidjson::kObjectType);
    auto& alloc = doc.GetAllocator();

    std::unordered_map<std::string, rapidjson::Value> sceneArrays;
    for (const Place& place : m_places)
    {
        rapidjson::Value entry(rapidjson::kObjectType);
        rapidjson::Value name; name.SetString(place.name.c_str(), static_cast<rapidjson::SizeType>(place.name.size()), alloc);
        entry.AddMember("name", name, alloc);
        entry.AddMember("position", MakeArray3(place.position, alloc), alloc);
        entry.AddMember("rotation", MakeArray4(place.rotation, alloc), alloc);
        entry.AddMember("fov", place.fov, alloc);

        auto it = sceneArrays.find(place.sceneName);
        if (it == sceneArrays.end())
            it = sceneArrays.emplace(place.sceneName, rapidjson::Value(rapidjson::kArrayType)).first;
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
        spdlog::error("Failed to write places.json");
}

void PlacesManager::SnapshotCamera()
{
    if (!m_camera) return;
    m_lastPos      = m_camera->GetPosition();
    m_lastRot      = m_camera->GetRotation();
    m_lastFov      = m_camera->GetFovYRadians();
    m_haveSnapshot = true;
}

bool PlacesManager::CameraDiffersFromSnapshot() const
{
    if (!m_camera) return false;

    const DirectX::XMFLOAT3& pos = m_camera->GetPosition();
    const DirectX::XMFLOAT4& rot = m_camera->GetRotation();
    const float              fov = m_camera->GetFovYRadians();

    return !FloatsEqual(pos.x, m_lastPos.x) || !FloatsEqual(pos.y, m_lastPos.y) || !FloatsEqual(pos.z, m_lastPos.z)
        || !FloatsEqual(rot.x, m_lastRot.x) || !FloatsEqual(rot.y, m_lastRot.y) || !FloatsEqual(rot.z, m_lastRot.z) || !FloatsEqual(rot.w, m_lastRot.w)
        || !FloatsEqual(fov,   m_lastFov);
}

size_t PlacesManager::ResolveGlobalIndex(size_t sceneIndex) const
{
    size_t seen = 0;
    for (size_t i = 0; i < m_places.size(); ++i)
    {
        if (m_places[i].sceneName != m_currentScene) continue;
        if (seen == sceneIndex) return i;
        ++seen;
    }
    return static_cast<size_t>(-1);
}
