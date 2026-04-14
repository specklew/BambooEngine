#pragma once

class Scene;
struct Primitive;
class AssetId;
class Renderer;

namespace ModelLoading
{
    std::shared_ptr<Scene> LoadScene(Renderer& renderer, const AssetId& assetId);
}
