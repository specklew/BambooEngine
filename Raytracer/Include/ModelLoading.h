#pragma once
#include "Primitive.h"

class Renderer;

namespace ModelLoading
{
    Primitive LoadModel(Renderer& renderer, const AssetId& assetId);
}
