#pragma once

#include <memory>

class Scene;
class AssetId;
class Renderer;

// Tungsten scene-format importer (ADR 0006). A second front-end to the engine's
// Scene, parallel to glTF: parses a Tungsten `.json`, loads its `.wo3`/`.obj`
// meshes into the canonical space (ADR 0005), maps BSDFs onto the metallic-
// roughness material, and lights the scene with the engine's default light
// (Tungsten emitters/env are not imported in v1).
namespace TungstenLoading
{
    std::shared_ptr<Scene> LoadScene(Renderer& renderer, const AssetId& assetId);
}
