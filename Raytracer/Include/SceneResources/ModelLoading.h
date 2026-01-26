#pragma once

class Scene;
struct Primitive;
class AssetId;
class Renderer;

namespace ModelLoading
{
    enum LOADED_SCENES
    {
        AVOCADO,
        A_BEAUTIFUL_GAME,
        SPONZA,
        SIMPLE_INSTANCING,
        DUCK,
    };

    inline std::vector<std::string> scenePaths = {
        "resources/models/avocado/avocado.glb",
        "resources/models/abeautifulgame.glb",
        "resources/models/sponza/gltf/sponza.gltf",
        "resources/models/simpleinstancing.glb",
        "resources/models/duck.glb",
    };

    inline std::string ScenePath(const LOADED_SCENES scene) { return scenePaths[scene]; }

    std::vector<std::shared_ptr<Scene>> LoadAllScenes(Renderer& renderer); 
    std::shared_ptr<Scene> LoadScene(Renderer& renderer, const AssetId& assetId);
}
