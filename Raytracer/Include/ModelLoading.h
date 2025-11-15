#pragma once

struct Primitive;
class AssetId;
class Renderer;

namespace ModelLoading
{
    std::shared_ptr<Primitive> LoadModel(Renderer& renderer, const AssetId& assetId);
    std::vector<std::shared_ptr<Primitive>> LoadFullModel(Renderer& renderer, const AssetId& assetId);
}
