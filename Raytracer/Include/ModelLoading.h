#pragma once

class Scene;
struct Primitive;
class AssetId;
class Renderer;

namespace ModelLoading
{
    enum LOADED_SCENES
    {
        A_BEAUTIFUL_GAME = 0,
        SPONZA = 1
    };

    inline std::vector<std::string> scenePaths = {
        "resources/models/abeautifulgame.glb",
        "resources/models/sponza/gltf/sponza.gltf"
    };

    std::vector<std::shared_ptr<Scene>> LoadAllScenes(Renderer& renderer); 
    
    std::shared_ptr<Primitive> LoadModel(Renderer& renderer, const AssetId& assetId);
    std::vector<std::shared_ptr<Primitive>> LoadFullModel(Renderer& renderer, const AssetId& assetId);
    std::shared_ptr<Scene> LoadScene(Renderer& renderer, const AssetId& assetId);
}
