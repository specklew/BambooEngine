#include "pch.h"
#include "Scene.h"

#include "SceneNode.h"

Scene::Scene()
{
    m_root = std::make_shared<SceneNode>();
}
