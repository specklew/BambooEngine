#pragma once
#include "Resources/StructuredBuffer.h"

class ConstantBuffer;
class VertexBuffer;
class IndexBuffer;
class AccelerationStructures;
class AssetId;
class Renderer;
struct Primitive;
class GameObject;
class Model;
class SceneNode;
class SceneBuilder;

struct GeometryInfo
{
    uint32_t vertexOffset;
    uint32_t indexOffset;
};

struct InstanceInfo
{
    uint32_t geometryId;
    int textureId;
};

class Scene
{
public:
    [[nodiscard]] std::shared_ptr<SceneNode> GetRoot() const { return m_root; }
    [[nodiscard]] const std::vector<std::shared_ptr<Model>>& GetModels() const { return m_models; }
    [[nodiscard]] const std::vector<std::shared_ptr<GameObject>>& GetGameObjects() const { return m_gameObjects; }
    [[nodiscard]] const std::string& GetName() const { return m_name; }
    [[nodiscard]] std::shared_ptr<AccelerationStructures> GetAccelerationStructures() { return m_rtRepresentation; }
    [[nodiscard]] std::shared_ptr<IndexBuffer> GetIndexBuffer() { return m_indexBuffer; }
    [[nodiscard]] std::shared_ptr<VertexBuffer> GetVertexBuffer() { return m_vertexBuffer; }
    [[nodiscard]] std::shared_ptr<StructuredBuffer<GeometryInfo>> GetGeometryInfoBuffer() { return m_geometryInfoBuffer; }
    [[nodiscard]] std::shared_ptr<StructuredBuffer<InstanceInfo>> GetInstanceInfoBuffer() { return m_instanceInfoBuffer; }
    
private:
    friend class SceneBuilder;
    Scene() = default;

    std::shared_ptr<IndexBuffer> m_indexBuffer;
    std::shared_ptr<VertexBuffer> m_vertexBuffer;
    std::shared_ptr<StructuredBuffer<GeometryInfo>> m_geometryInfoBuffer;
    std::shared_ptr<StructuredBuffer<InstanceInfo>> m_instanceInfoBuffer;
    
    std::string m_name;
    std::shared_ptr<SceneNode> m_root;
    std::vector<std::shared_ptr<GameObject>> m_gameObjects;
    std::vector<std::shared_ptr<Model>> m_models;

    std::shared_ptr<AccelerationStructures> m_rtRepresentation;
};

class SceneBuilder
{
public:
    SceneBuilder();
    void AddGameObject(const std::shared_ptr<GameObject>& gameObject, const std::shared_ptr<Model>& model);
    void AddModel(const std::shared_ptr<Model>& model);
    void AddChild(const std::shared_ptr<SceneNode>& parent, const std::shared_ptr<SceneNode>& child);
    void SetName(const std::string& name);
    void SetAccelerationStructures(const std::shared_ptr<AccelerationStructures>& accelerationStructures);
    void SetVertexBuffer(const std::shared_ptr<VertexBuffer>& vertexBuffer);
    void SetIndexBuffer(const std::shared_ptr<IndexBuffer>& indexBuffer);
    void UpdateMatrices();

    std::shared_ptr<SceneNode> GetRoot() const { return m_root; }
    std::vector<std::shared_ptr<GameObject>> GetGameObjects() const { return m_gameObjects; }
    std::shared_ptr<Model> GetModel(const int index) const { return m_models[index]; }
    std::vector<std::shared_ptr<Model>> GetModels() const { return m_models; }
    
    std::shared_ptr<Scene> Build(Renderer& renderer);
    
private:
    friend class SceneNode;

    static void UpdateMatricesInNodesRecursively(const std::shared_ptr<SceneNode>& node);
    std::vector<std::shared_ptr<Primitive>> GetAllPrimitives() const; 

    std::shared_ptr<IndexBuffer> m_indexBuffer;
    std::shared_ptr<VertexBuffer> m_vertexBuffer;
    
    std::string m_name;
    std::vector<std::shared_ptr<GameObject>> m_gameObjects;
    std::vector<std::shared_ptr<Model>> m_models;
    std::shared_ptr<SceneNode> m_root;

    std::shared_ptr<AccelerationStructures> m_rtRepresentation;

    bool m_isBuilt = false;
};