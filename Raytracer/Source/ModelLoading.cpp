#include "pch.h"
#include "ModelLoading.h"

#include <filesystem>
#include <tinygltf/tiny_gltf.h>
#include <spdlog/spdlog.h>

#include "InputElements.h"
#include "Model.h"
#include "Primitive.h"
#include "Renderer.h"
#include "Scene.h"
#include "SceneNode.h"
#include "ResourceManager/ResourceManagerTypes.h"
#include "tinygltf/tiny_gltf.h"

static void ExtractVertices(tinygltf::Model& model, std::vector<Vertex>& outVertices)
{
    tinygltf::Primitive& primitive = model.meshes[0].primitives[0];

    const bool has_position = primitive.attributes.find("POSITION") != primitive.attributes.end();
    assert(has_position && "Mesh primitive must have POSITION attribute");

    const float* positions = nullptr;
    size_t vertex_count = 0;
    
    // POSITION
    {
        const tinygltf::Accessor& accessor = model.accessors[primitive.attributes["POSITION"]];
        const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

        positions = reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
        assert(positions && "Failed to extract vertices from glTF model. The POSITION attribute is null.");
        
        // Each accessor for attributes has the same count.
        vertex_count = accessor.count;
    }

    // PROCESS

    for (size_t i = 0; i < vertex_count; i++)
    {
        Vertex vertex{};

        // Inverting Y and Z to convert from right-handed to left-handed coordinate system. https://stackoverflow.com/questions/16986017/how-do-i-make-blender-operate-in-left-hand
        vertex.Pos.x = positions[i * 3 + 0];
        vertex.Pos.y = positions[i * 3 + 1];
        vertex.Pos.z = positions[i * 3 + 2];

        outVertices.push_back(vertex);
    }
}

static void ExtractVertices(const tinygltf::Model& model, tinygltf::Primitive& primitive, std::vector<Vertex>& outVertices)
{
    const bool has_position = primitive.attributes.find("POSITION") != primitive.attributes.end();
    assert(has_position && "Mesh primitive must have POSITION attribute");

    const float* positions = nullptr;
    size_t vertex_count = 0;
    
    // POSITION
    {
        const tinygltf::Accessor& accessor = model.accessors[primitive.attributes["POSITION"]];
        const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

        positions = reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
        assert(positions && "Failed to extract vertices from glTF model. The POSITION attribute is null.");
        
        // Each accessor for attributes has the same count.
        vertex_count = accessor.count;
    }

    // PROCESS

    for (size_t i = 0; i < vertex_count; i++)
    {
        Vertex vertex{};

        // Inverting Y and Z to convert from right-handed to left-handed coordinate system. https://stackoverflow.com/questions/16986017/how-do-i-make-blender-operate-in-left-hand
        vertex.Pos.x = positions[i * 3 + 0];
        vertex.Pos.y = positions[i * 3 + 1];
        vertex.Pos.z = positions[i * 3 + 2];

        outVertices.push_back(vertex);
    }
}

static void ExtractIndices(tinygltf::Model& model, std::vector<uint32_t>& outIndices)
{
    tinygltf::Primitive& primitive = model.meshes[0].primitives[0];

    assert(primitive.indices >= 0 && "Failed loading glTF model. Mesh primitive must have indices");

    const auto& accessor = model.accessors[primitive.indices];
    const auto& buffer_view = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[buffer_view.buffer];
    
    switch (accessor.componentType)
    {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            const uint16_t* rawIndices = reinterpret_cast<const uint16_t*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
            outIndices.reserve(accessor.count);
            for (int i = 0; i < accessor.count; i+=3)
            {
                outIndices.push_back(rawIndices[i]);
                outIndices.push_back(rawIndices[i+1]);
                outIndices.push_back(rawIndices[i+2]);
            }

            break;
        }

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        {
            const uint32_t* rawIndices = reinterpret_cast<const uint32_t*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
            outIndices.reserve(accessor.count);
            for (int i = 0; i < accessor.count; i+=3)
            {
                outIndices.push_back(rawIndices[i]);
                outIndices.push_back(rawIndices[i+1]);
                outIndices.push_back(rawIndices[i+2]);
            }
            break;
        }
        default:
        {
            assert("False: Unsupported index component type");
            break;            
        }
    }
}

static void ExtractIndices(const tinygltf::Model& model, const tinygltf::Primitive& primitive, std::vector<uint32_t>& outIndices)
{
    assert(primitive.indices >= 0 && "Failed loading glTF model. Mesh primitive must have indices");

    const auto& accessor = model.accessors[primitive.indices];
    const auto& buffer_view = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[buffer_view.buffer];
    
    switch (accessor.componentType)
    {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            const uint16_t* rawIndices = reinterpret_cast<const uint16_t*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
            outIndices.reserve(accessor.count);
            for (int i = 0; i < accessor.count; i+=3)
            {
                outIndices.push_back(rawIndices[i]);
                outIndices.push_back(rawIndices[i+1]);
                outIndices.push_back(rawIndices[i+2]);
            }

            break;
        }

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        {
            const uint32_t* rawIndices = reinterpret_cast<const uint32_t*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
            outIndices.reserve(accessor.count);
            for (int i = 0; i < accessor.count; i+=3)
            {
                outIndices.push_back(rawIndices[i]);
                outIndices.push_back(rawIndices[i+1]);
                outIndices.push_back(rawIndices[i+2]);
            }
            break;
        }
        default:
        {
            assert("False: Unsupported index component type");
            break;            
        }
    }
}

static bool LoadTinyGLTFModel(const std::filesystem::path &path, tinygltf::Model& outModel)
{
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    spdlog::debug("Loading glTF model from path: {}", path.string());
    
    bool success;
    if (path.extension() == ".glb")
    {
        success = loader.LoadBinaryFromFile(&outModel, &err, &warn, path.string());
    }
    else
    {
        success = loader.LoadASCIIFromFile(&outModel, &err, &warn, path.string());
    }
    
    if (!warn.empty()) {
        spdlog::warn("TinyGLTF warning: {}", warn);
    }
    if (!err.empty()) {
        spdlog::error("TinyGLTF error: {}", err);
    }
    if (!success) {
        spdlog::error("Failed to load glTF model: {}", path.string());
        return false;
    }

    return true;
}

static std::shared_ptr<Primitive> LoadPrimitive(Renderer& renderer, const tinygltf::Model& model, tinygltf::Primitive& primitive)
{
    assert(primitive.indices >= 0 && "Failed loading glTF model. Mesh primitive must have indices");

    std::vector<Vertex> vertices = {};
    std::vector<uint32_t> indices = {};
    
    ExtractIndices(model, primitive, indices);
    ExtractVertices(model, primitive, vertices);

    assert(vertices.size() < INT_MAX);
    assert(indices.size() < INT_MAX);
    
    // Here we would normally calculate tangents and AABB.

    return renderer.CreatePrimitive(vertices, indices);
}

std::shared_ptr<Primitive> ModelLoading::LoadModel(Renderer& renderer, const AssetId& assetId)
{
    tinygltf::Model model;
    bool succeeded = LoadTinyGLTFModel(assetId.AsPath(), model);

    assert(succeeded && "Failed to load model");
    
    // TODO: Support more.
    if (model.meshes.size() != 1)
    {
        spdlog::error("The engine is currently limited to loading models with exactly one mesh. The model at path {} has {} meshes.", assetId.AsString(), model.meshes.size());
        assert(false && "Failed to load model. Only one mesh per model is supported.");
    }

    if (model.meshes[0].primitives.size() != 1)
    {
        spdlog::error("The engine is currently limited to loading models with exactly one primitive per mesh. The model at path {} has {} primitives in its first mesh.", assetId.AsString(), model.meshes[0].primitives.size());
        assert(false && "Failed to load model. Only one primitive per mesh is supported.");
    }

    std::vector<Vertex> vertices = {};
    std::vector<uint32_t> indices = {};

    ExtractVertices(model, vertices);
    ExtractIndices(model, indices);
    
    assert(vertices.size() < INT_MAX);
    assert(indices.size() < INT_MAX);
    
    // Here we would normally calculate tangents and AABB.

    return renderer.CreatePrimitive(vertices, indices);
}

std::vector<std::shared_ptr<Primitive>> ModelLoading::LoadFullModel(Renderer& renderer, const AssetId& assetId)
{
    tinygltf::Model model;
    bool succeeded = LoadTinyGLTFModel(assetId.AsPath(), model);

    assert(succeeded && "Failed to load model");
    
    std::vector<std::shared_ptr<Primitive>> primitives;
    
    for (auto& mesh : model.meshes)
    {
        for (auto& primitive : mesh.primitives)
        {
            auto prim = LoadPrimitive(renderer, model, primitive);
            primitives.push_back(prim);
        }
    }

    spdlog::info("Loaded model at path {} with {} meshes and {} primitives.", assetId.AsString(), model.meshes.size(), primitives.size());
    
    return primitives;
}

std::shared_ptr<Scene> ModelLoading::LoadScene(Renderer& renderer, const AssetId& assetId)
{
    tinygltf::Model model;
    bool succeeded = LoadTinyGLTFModel(assetId.AsPath(), model);

    assert(succeeded && "Failed to load model");
    assert(model.scenes.size() > 0 && "Model has no scenes!");
    
    auto scene = std::make_shared<Scene>();
    scene->Initialize(); // TODO: I hate this initialization, but shared_from_this needs to be called after something that happens during initialization.
    auto sceneRoot = scene->GetRoot();
    
    auto gltf_scene = model.scenes[model.defaultScene];

    spdlog::debug("Scene nodes in glTF scene: {}", gltf_scene.nodes.size());
    spdlog::debug("Model nodes in glTF model: {}", model.nodes.size());
    
    // TODO: Make this part recursive and extracted!!
    for (auto gltf_node_index : gltf_scene.nodes)
    {
        std::shared_ptr<SceneNode> current_node = std::make_shared<SceneNode>();
        sceneRoot->AddChild(current_node);
        
        auto gltf_node = model.nodes[gltf_node_index];

        if (gltf_node.mesh != 0)
        {
            auto node_mesh = model.meshes[gltf_node.mesh];
            auto current_model = renderer.InstantiateModel();
            
            for (auto& primitive : node_mesh.primitives)
            {
                auto prim = LoadPrimitive(renderer, model, primitive);
                current_model->AddMesh(prim);
            }

            current_node->AddModel(current_model);
        }

        if (gltf_node.translation.size() > 0)
        {
            // May cause errors, but I'm not changing float precision to double.
            float x = gltf_node.translation[0];
            float y = gltf_node.translation[1];
            float z = gltf_node.translation[2];

            current_node->SetPosition({x, y, z});
        }

        if (gltf_node.rotation.size() > 0)
        {

            float x = gltf_node.rotation[0];
            float y = gltf_node.rotation[1];
            float z = gltf_node.rotation[2];
            float w = gltf_node.rotation[3];
            current_node->SetRotation({x, y, z, w});
        }

        if (gltf_node.scale.size() > 0)
        {

            float x = gltf_node.scale[0];
            float y = gltf_node.scale[1];
            float z = gltf_node.scale[2];
            current_node->SetScale({x, y, z});
        }
    }
    // END - TODO

    return scene;
}
