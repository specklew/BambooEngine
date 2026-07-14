#pragma once

#include <functional>
#include <string>
#include <vector>

#include <DirectXMath.h>

#include "SceneResources/LightData.h"

class Camera;

struct State
{
    std::string sceneName;
    std::string name;

    DirectX::XMFLOAT3 position{ 0.0f, 0.0f, 0.0f };
    DirectX::XMFLOAT4 rotation{ 0.0f, 0.0f, 0.0f, 1.0f };

    float fov = 0.0f;

    bool hasLights = false;            // false = camera-only (no `lights` field on disk)
    std::vector<LightData> lights;     // valid only when hasLights == true
};

class StatesManager
{
public:
    void Load();
    void SetCamera(Camera& camera) { m_camera = &camera; }

    void SetLightsAccessors(
        std::function<std::vector<LightData>()>            getLights,
        std::function<void(const std::vector<LightData>&)> setLights)
    {
        m_getLights = std::move(getLights);
        m_setLights = std::move(setLights);
    }

    void OnSceneChanged(const std::string& modelName);
    void Tick();

    const std::string& GetCurrentScene()    const { return m_currentScene; }
    const std::string& GetActiveStateName() const { return m_activeState; }
    bool HasCurrentScene()    const { return !m_currentScene.empty(); }
    std::vector<State> GetStatesForCurrentScene() const;

    bool AddStateFromScene();
    bool RenameState(size_t sceneIndex, const std::string& newName);
    void DeleteState(size_t sceneIndex);
    void GoToState(size_t sceneIndex);
    bool GoToStateByName(const std::string& name);

private:
    static constexpr const char* kSavePath = "SavedUserData/states.json";

    bool IsNameTakenInCurrentScene(const std::string& name) const;
    std::string GenerateUniqueName(const std::string& base) const;

    void ParseInto(const std::string& buffer, std::vector<State>& out) const;
    void Save() const;
    void SnapshotCamera();
    bool CameraDiffersFromSnapshot() const;

    size_t ResolveGlobalIndex(size_t sceneIndex) const;

    std::vector<State> m_states;
    Camera* m_camera = nullptr;
    std::function<std::vector<LightData>()>            m_getLights;
    std::function<void(const std::vector<LightData>&)> m_setLights;
    std::string m_currentScene;
    std::string m_activeState;

    DirectX::XMFLOAT3 m_lastPos{};
    DirectX::XMFLOAT4 m_lastRot{};
    float m_lastFov = 0.0f;
    bool m_haveSnapshot = false;
};
