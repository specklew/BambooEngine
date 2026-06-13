#pragma once

#include <string>
#include <vector>

#include <DirectXMath.h>

class Camera;

struct Place
{
    std::string sceneName;
    std::string name;

    DirectX::XMFLOAT3 position{ 0.0f, 0.0f, 0.0f };
    DirectX::XMFLOAT4 rotation{ 0.0f, 0.0f, 0.0f, 1.0f };

    float fov = 0.0f;
};

class PlacesManager
{
public:
    void Load();
    void SetCamera(Camera& camera) { m_camera = &camera; }

    void OnSceneChanged(const std::string& modelName);
    void Tick();

    const std::string& GetCurrentScene()    const { return m_currentScene; }
    const std::string& GetActivePlaceName() const { return m_activePlace; }
    bool HasCurrentScene()    const { return !m_currentScene.empty(); }
    std::vector<Place> GetPlacesForCurrentScene() const;

    bool AddPlaceFromCamera();
    bool RenamePlace(size_t sceneIndex, const std::string& newName);
    void DeletePlace(size_t sceneIndex);
    void GoToPlace(size_t sceneIndex);
    bool GoToPlaceByName(const std::string& name);

private:
    static constexpr const char* kSavePath = "SavedUserData/places.json";

    bool IsNameTakenInCurrentScene(const std::string& name) const;
    std::string GenerateUniqueName(const std::string& base) const;

    void Save() const;
    void SnapshotCamera();
    bool CameraDiffersFromSnapshot() const;

    size_t ResolveGlobalIndex(size_t sceneIndex) const;

    std::vector<Place> m_places;
    Camera* m_camera = nullptr;
    std::string m_currentScene;
    std::string m_activePlace;

    DirectX::XMFLOAT3 m_lastPos{};
    DirectX::XMFLOAT4 m_lastRot{};
    float m_lastFov = 0.0f;
    bool m_haveSnapshot = false;
};
