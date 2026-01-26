#include "pch.h"
#include "SceneResources/ModelLoading.h"

#include <filesystem>
#include <tinygltf/tiny_gltf.h>
#include <spdlog/spdlog.h>

#include "AccelerationStructures.h"
#include "InputElements.h"
#include "SceneResources/Model.h"
#include "SceneResources/Primitive.h"
#include "Renderer.h"
#include "SceneResources/Scene.h"
#include "SceneResources/SceneNode.h"
#include "ResourceManager/ResourceManagerTypes.h"
#include "Resources/IndexBuffer.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/GameObject.h"
#include "SceneResources/Material.h"
#include "tinygltf/tiny_gltf.h"

static void ExtractVertices(const tinygltf::Model& model, tinygltf::Primitive& primitive, std::vector<Vertex>& outVertices)
{
    const bool has_position = primitive.attributes.find("POSITION") != primitive.attributes.end();
    const bool has_normal = primitive.attributes.find("NORMAL") != primitive.attributes.end();
    const bool has_tex_coords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
    assert(has_position && "Mesh primitive must have POSITION attribute");

    const float* positions = nullptr;
    const float* texCoords = nullptr;
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

    //TEXTURE COORDINATES
    if (has_tex_coords)
    {
        const tinygltf::Accessor& accessor = model.accessors[primitive.attributes["TEXCOORD_0"]];
        const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

        texCoords = reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
    }

    // PROCESS

    for (size_t i = 0; i < vertex_count; i++)
    {
        Vertex vertex{};

        // Inverting Y and Z to convert from right-handed to left-handed coordinate system. https://stackoverflow.com/questions/16986017/how-do-i-make-blender-operate-in-left-hand
        vertex.Pos.x = positions[i * 3 + 0];
        vertex.Pos.y = positions[i * 3 + 1];
        vertex.Pos.z = positions[i * 3 + 2];

        vertex.Normal = DirectX::XMFLOAT3{0, 1, 0};
        vertex.Tex0 = DirectX::XMFLOAT2{0,0};

        if (has_tex_coords)
        {
            vertex.Tex0.x = texCoords[i * 2 + 0];
            vertex.Tex0.y = texCoords[i * 2 + 1];
        }

        outVertices.push_back(vertex);
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
                outIndices.push_back(rawIndices[i+2]);
                outIndices.push_back(rawIndices[i+1]);
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
                outIndices.push_back(rawIndices[i+2]);
                outIndices.push_back(rawIndices[i+1]);
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

    std::shared_ptr<Material> material = std::make_shared<Material>();

    if (!model.materials.empty())
    {
        if (int albedo_index = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index; albedo_index >= 0)
        {
            const tinygltf::Image albedoImage = model.images[model.textures[albedo_index].source];
            material->m_albedoTexture = renderer.CreateTextureFromGLTF(albedoImage);
            material->UpdateMaterial();
        }
    }
    
    return renderer.CreatePrimitive(vertices, indices, material);
}

std::vector<std::shared_ptr<Scene>> ModelLoading::LoadAllScenes(Renderer& renderer)
{
    std::vector<std::shared_ptr<Scene>> scenes;
    
    for (const std::string& scenePath : scenePaths)
    {
        scenes.push_back(LoadScene(renderer, AssetId(scenePath.c_str(), scenePath.size())));
    }

    return std::move(scenes);
}

static DirectX::SimpleMath::Vector3 ReadNodePosition(const tinygltf::Node& node)
{
    if (node.translation.size() == 3)
    {
        return {
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2])
        };
    }
    return {0.0f, 0.0f, 0.0f};
}

static DirectX::SimpleMath::Quaternion ReadNodeRotation(const tinygltf::Node& node)
{
    if (node.rotation.size() == 4)
    {
        return {
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2]),
            static_cast<float>(node.rotation[3])
        };
    }
    return {0.0f, 0.0f, 0.0f, 1.0f};
}

static DirectX::SimpleMath::Vector3 ReadNodeScale(const tinygltf::Node& node)
{
    if (node.scale.size() == 3)
    {
        return {
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2])
        };
    }
    return {1.0f, 1.0f, 1.0f};
}

static void TraverseNode(Renderer& renderer, const tinygltf::Model& model, SceneBuilder& sceneBuilder, int nodeIndex, std::shared_ptr<SceneNode> parentNode)
{
    const tinygltf::Node& gltf_node = model.nodes[nodeIndex];
    auto current_node = std::make_shared<SceneNode>();
    
    if (gltf_node.mesh > 0)
    {
        auto gameObject = renderer.InstantiateGameObject();
        sceneBuilder.AddGameObject(gameObject, sceneBuilder.GetModel(gltf_node.mesh));
        current_node->AddGameObject(gameObject);
    }

    current_node->SetPosition(ReadNodePosition(gltf_node));
    current_node->SetRotation(ReadNodeRotation(gltf_node));
    current_node->SetScale(ReadNodeScale(gltf_node));

    sceneBuilder.AddChild(parentNode, current_node);
    
    for (auto child_index : gltf_node.children)
    {
        TraverseNode(renderer, model, sceneBuilder, child_index, current_node);
    }
}

static std::shared_ptr<AccelerationStructures> BuildAccelerationStructures(const Renderer& renderer, const SceneBuilder& scene)
{
    using Microsoft::WRL::ComPtr;
    
    std::unordered_map<std::shared_ptr<Model>, std::shared_ptr<AccelerationStructureBuffers>> modelBLASes;
    std::shared_ptr<AccelerationStructures> as = std::make_shared<AccelerationStructures>();

    for (auto model : scene.GetModels())
    {
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vertex_buffers;
        std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> index_buffers;
		
        for (auto prim : model->GetMeshes())
        {

            std::pair<ComPtr<ID3D12Resource>, uint32_t> vertex_pair = {
                prim->GetVertexBuffer()->GetUnderlyingResource(),
                prim->GetVertexBuffer()->GetVertexCount()
            };
			
            std::pair<ComPtr<ID3D12Resource>, uint32_t> index_pair = {
                prim->GetIndexBuffer()->GetUnderlyingResource(),
                prim->GetIndexBuffer()->GetIndexCount()
            };
			
            vertex_buffers.emplace_back(vertex_pair);
            index_buffers.emplace_back(index_pair);
        }

        AccelerationStructureBuffers bottomLevelBuffers = as->CreateBottomLevelAS(
            renderer.g_d3d12Device.Get(),
            renderer.GetCommandList(),
            vertex_buffers,
            index_buffers);

        modelBLASes.emplace(model, std::make_shared<AccelerationStructureBuffers>(bottomLevelBuffers));
    }
    
    for (auto go : scene.GetGameObjects())
    {
        auto model = go->GetModel();
        auto blasIt = modelBLASes.find(model);
        if (blasIt != modelBLASes.end())
        {
            DirectX::XMMATRIX worldMatrix = DirectX::XMLoadFloat4x4(&go->GetWorldFloat4X4());
            auto instance = std::pair(blasIt->second->p_result, worldMatrix);
            as->GetInstances().emplace_back(instance);
        }
        else
        {
            spdlog::warn("Model for GameObject not found in BLASes map during TLAS setup.");
        }
    }

    as->CreateTopLevelAS(Renderer::g_d3d12Device.Get(), renderer.GetCommandList().Get(), as->GetInstances(), false);

    return as;
}

std::shared_ptr<Scene> ModelLoading::LoadScene(Renderer& renderer, const AssetId& assetId)
{
    spdlog::info("Loading scene {}", assetId.AsString());
    
    tinygltf::Model model;
    bool succeeded = LoadTinyGLTFModel(assetId.AsPath(), model);

    assert(succeeded && "Failed to load model");
    assert(model.scenes.size() > 0 && "Model has no scenes!");
    
    auto scene_builder = SceneBuilder();

    scene_builder.SetName(assetId.AsString());
    
    for (auto gltf_model : model.meshes)
    {
        auto current_model = std::make_shared<Model>();
        for (auto& primitive : gltf_model.primitives)
        {
            auto prim = LoadPrimitive(renderer, model, primitive);
            current_model->AddMesh(prim);
        }
        scene_builder.AddModel(current_model);
    }

    if (model.defaultScene < 0)
    {
        spdlog::warn("Failed to load defaultScene from model: {}, defaulting to 0.", assetId.AsString());
        model.defaultScene = 0;
    }
    
    auto gltf_scene = model.scenes[model.defaultScene];

    spdlog::debug("Scene nodes in glTF scene: {}", gltf_scene.nodes.size());
    spdlog::debug("Model nodes in glTF model: {}", model.nodes.size());

    for (auto gltf_node_index : gltf_scene.nodes)
    {
        TraverseNode(renderer, model, scene_builder, gltf_node_index, scene_builder.GetRoot());
    }

    // Fallback behaviour
    if (scene_builder.GetGameObjects().empty())
    {
        if (model.nodes.size() == 1)
        {
            scene_builder.GetRoot()->SetPosition(ReadNodePosition(model.nodes[0]));
            scene_builder.GetRoot()->SetRotation(ReadNodeRotation(model.nodes[0]));
            scene_builder.GetRoot()->SetScale(ReadNodeScale(model.nodes[0]));
        }
        
        int model_index = 0;
        for (auto gltfModel : model.meshes)
        {
            auto current_node = std::make_shared<SceneNode>();
            auto gameObject = renderer.InstantiateGameObject();
            scene_builder.AddGameObject(gameObject, scene_builder.GetModel(model_index));
            current_node->AddGameObject(gameObject);

            scene_builder.AddChild(scene_builder.GetRoot(), current_node);
            
            model_index++;
        }
    }

    scene_builder.UpdateMatrices(); // Needs to be done before TLAS building for instances to have the correct positions.
    scene_builder.SetAccelerationStructures(BuildAccelerationStructures(renderer, scene_builder));
    
    return scene_builder.Build();
} 