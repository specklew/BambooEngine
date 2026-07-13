#include "pch.h"
#include "SceneResources/ModelLoading.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <tinygltf/tiny_gltf.h>
#include <spdlog/spdlog.h>

#include "AccelerationStructures.h"
#include "InputElements.h"
#include "SceneResources/MeshProcessing.h"
#include "SceneResources/TungstenLoading.h"
#include "SceneResources/Model.h"
#include "SceneResources/Primitive.h"
#include "Renderer.h"
#include "SceneResources/Scene.h"
#include "SceneResources/SceneNode.h"
#include "ResourceManager/ResourceManagerTypes.h"
#include "SceneResources/GameObject.h"
#include "SceneResources/Material.h"
#include "Resources/IndexBuffer.h"
#include "Resources/VertexBuffer.h"

static void ExtractVertices(const tinygltf::Model& model, tinygltf::Primitive& primitive, std::vector<Vertex>& outVertices)
{
    const bool has_position = primitive.attributes.find("POSITION") != primitive.attributes.end();
    const bool has_normal = primitive.attributes.find("NORMAL") != primitive.attributes.end();
    const bool has_tangent = primitive.attributes.find("TANGENT") != primitive.attributes.end();
    const bool has_tex_coords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
    assert(has_position && "Mesh primitive must have POSITION attribute");

    const float* positions = nullptr;
    const float* normals = nullptr;
    const float* tangents = nullptr;
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

    if (has_normal)
    {
        const tinygltf::Accessor& accessor = model.accessors[primitive.attributes["NORMAL"]];
        const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

        normals = reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
    }

    if (has_tangent)
    {
        const tinygltf::Accessor& accessor = model.accessors[primitive.attributes["TANGENT"]];
        const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

        tangents = reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);
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

        if (has_normal)
        {
            vertex.Normal.x = normals[i * 3 + 0];
            vertex.Normal.y = normals[i * 3 + 1];
            vertex.Normal.z = normals[i * 3 + 2];
        }
        
        if (has_tex_coords)
        {
            vertex.Tex0.x = texCoords[i * 2 + 0];
            vertex.Tex0.y = texCoords[i * 2 + 1];
        }

        if (has_tangent)
        {
            // glTF TANGENT is VEC4: xyz = tangent direction, w = handedness (+/-1)
            vertex.Tangent.x = tangents[i * 4 + 0];
            vertex.Tangent.y = tangents[i * 4 + 1];
            vertex.Tangent.z = tangents[i * 4 + 2];
            vertex.Tangent.w = tangents[i * 4 + 3];
        }
        else
        {
            vertex.Tangent = {0, 0, 0, 1.0f};  // Accumulated per-triangle in ComputeTangents
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

static std::shared_ptr<Texture> GetOrCreateTexture(Renderer& renderer, const tinygltf::Model& model, int textureIndex, std::unordered_map<int, std::shared_ptr<Texture>>& textureCache)
{
    int sourceIndex = model.textures[textureIndex].source;
    auto it = textureCache.find(sourceIndex);
    if (it != textureCache.end())
        return it->second;

    auto texture = renderer.CreateTextureFromGLTF(model.images[sourceIndex]);
    textureCache[sourceIndex] = texture;
    return texture;
}

static std::shared_ptr<Primitive> LoadPrimitive(Renderer& renderer, const tinygltf::Model& model, tinygltf::Primitive& primitive, std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices, std::unordered_map<int, std::shared_ptr<Texture>>& textureCache)
{
    assert(primitive.indices >= 0 && "Failed loading glTF model. Mesh primitive must have indices");

    std::vector<Vertex> vertices = {};
    std::vector<uint32_t> indices = {};
    
    ExtractIndices(model, primitive, indices);
    ExtractVertices(model, primitive, vertices);

    assert(vertices.size() < INT_MAX);
    assert(indices.size() < INT_MAX);

    MeshUtils::DropDegenerateTriangles(vertices, indices);

    const bool has_normal = primitive.attributes.find("NORMAL") != primitive.attributes.end();
    if (!has_normal)
        MeshUtils::ComputeNormals(vertices, indices);

    const bool has_tangent = primitive.attributes.find("TANGENT") != primitive.attributes.end();
    if (!has_tangent)
        MeshUtils::ComputeTangents(vertices, indices);

    MeshUtils::EnforceVertexInvariants(vertices);

    std::shared_ptr<Material> material = std::make_shared<Material>();

    if (!model.materials.empty())
    {
        if (int albedo_index = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index; albedo_index >= 0)
        {
            material->m_albedoTexture = GetOrCreateTexture(renderer, model, albedo_index, textureCache);
        }

        if (int normal_texture_index = model.materials[primitive.material].normalTexture.index; normal_texture_index >= 0)
        {
            material->m_normalTexture = GetOrCreateTexture(renderer, model, normal_texture_index, textureCache);
        }

        if (int metallic_roughness_index = model.materials[primitive.material].pbrMetallicRoughness.metallicRoughnessTexture.index; metallic_roughness_index >= 0)
        {
            material->m_metallicRoughnessTexture = GetOrCreateTexture(renderer, model, metallic_roughness_index, textureCache);
        }
        
        if (model.materials[primitive.material].alphaMode != "OPAQUE")
        {
            material->m_data.isOpaque = false;
        }

        const auto& pbr = model.materials[primitive.material].pbrMetallicRoughness;
        const auto& bcf = pbr.baseColorFactor;
        material->m_data.baseColorFactor = { static_cast<float>(bcf[0]), static_cast<float>(bcf[1]), static_cast<float>(bcf[2]), static_cast<float>(bcf[3]) };
        material->m_data.metallicFactor = static_cast<float>(pbr.metallicFactor);
        material->m_data.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

        material->UpdateMaterial();
    }

    auto index_view = BufferView();
    index_view.buffer = nullptr;
    index_view.count = indices.size();
    index_view.offset = outIndices.size();
    index_view.offsetBytes = outIndices.size() * sizeof(uint32_t);
    index_view.size = indices.size() * sizeof(uint32_t);

    auto vertex_view = BufferView();
    vertex_view.buffer = nullptr;
    vertex_view.count = vertices.size();
    vertex_view.offset = outVertices.size();
    vertex_view.offsetBytes = outVertices.size() * sizeof(Vertex);
    vertex_view.size = vertices.size() * sizeof(Vertex);

    DirectX::XMFLOAT3 local_min{  FLT_MAX,  FLT_MAX,  FLT_MAX };
    DirectX::XMFLOAT3 local_max{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (const Vertex& v : vertices)
    {
        local_min.x = std::min(local_min.x, v.Pos.x);
        local_min.y = std::min(local_min.y, v.Pos.y);
        local_min.z = std::min(local_min.z, v.Pos.z);
        local_max.x = std::max(local_max.x, v.Pos.x);
        local_max.y = std::max(local_max.y, v.Pos.y);
        local_max.z = std::max(local_max.z, v.Pos.z);
    }

    // Need to be done here after buffer views to have correct offset.
    outVertices.insert(outVertices.end(), vertices.begin(), vertices.end());
    outIndices.insert(outIndices.end(), indices.begin(), indices.end());

    auto prim = std::make_shared<Primitive>(vertex_view, index_view, material);
    prim->m_localAabbMin = local_min;
    prim->m_localAabbMax = local_max;
    return prim;
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

static DirectX::XMMATRIX ComputeNodeLocalMatrix(const tinygltf::Node& node);

static void TraverseNode(Renderer& renderer, const tinygltf::Model& model, SceneBuilder& sceneBuilder, int nodeIndex, std::shared_ptr<SceneNode> parentNode)
{
    const tinygltf::Node& gltf_node = model.nodes[nodeIndex];
    auto current_node = std::make_shared<SceneNode>();

    if (gltf_node.mesh >= 0)
    {
        auto gameObject = renderer.InstantiateGameObject();
        sceneBuilder.AddGameObject(gameObject, sceneBuilder.GetModel(gltf_node.mesh));
        current_node->AddGameObject(gameObject);
    }

    if (gltf_node.matrix.size() == 16)
    {
        // glTF nodes may carry their transform as a raw matrix
        DirectX::XMVECTOR scale, rotation, translation;
        if (DirectX::XMMatrixDecompose(&scale, &rotation, &translation,
                ComputeNodeLocalMatrix(gltf_node)))
        {
            current_node->SetPosition(DirectX::SimpleMath::Vector3(translation));
            current_node->SetRotation(DirectX::SimpleMath::Quaternion(rotation));
            current_node->SetScale(DirectX::SimpleMath::Vector3(scale));
        }
        else
        {
            spdlog::warn("glTF node '{}' has a non-decomposable matrix (mirror/shear); transform dropped",
                gltf_node.name);
        }
    }
    else
    {
        current_node->SetPosition(ReadNodePosition(gltf_node));
        current_node->SetRotation(ReadNodeRotation(gltf_node));
        current_node->SetScale(ReadNodeScale(gltf_node));
    }

    sceneBuilder.AddChild(parentNode, current_node);
    
    for (auto child_index : gltf_node.children)
    {
        TraverseNode(renderer, model, sceneBuilder, child_index, current_node);
    }
}

static DirectX::XMMATRIX ComputeNodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        float m[16];
        for (int i = 0; i < 16; ++i)
            m[i] = static_cast<float>(node.matrix[i]);
        return DirectX::XMMATRIX(m);
    }

    auto translation = ReadNodePosition(node);
    auto rotation = ReadNodeRotation(node);
    auto scale = ReadNodeScale(node);

    return DirectX::XMMatrixAffineTransformation(
        DirectX::XMLoadFloat3(&scale),
        DirectX::g_XMZero,
        DirectX::XMLoadFloat4(&rotation),
        DirectX::XMLoadFloat3(&translation));
}

static void CollectLightsFromNode(
    const tinygltf::Model& model,
    int nodeIndex,
    DirectX::XMMATRIX parentWorldMatrix,
    const std::vector<LightData>& lightTemplates,
    std::vector<LightData>& outLights)
{
    const tinygltf::Node& node = model.nodes[nodeIndex];
    DirectX::XMMATRIX worldMatrix = ComputeNodeLocalMatrix(node) * parentWorldMatrix;

    if (node.light >= 0)
    {
        LightData light = lightTemplates.at(node.light);

        DirectX::XMVECTOR pos = DirectX::XMVector3Transform(
            DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f), worldMatrix);
        DirectX::XMStoreFloat3(&light.position, pos);

        // glTF lights emit along local -Z
        DirectX::XMVECTOR dir = DirectX::XMVector3Normalize(DirectX::XMVector3TransformNormal(
            DirectX::XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f), worldMatrix));
        DirectX::XMStoreFloat3(&light.direction, dir);

        outLights.emplace_back(light);
    }

    for (int childIndex : node.children)
    {
        CollectLightsFromNode(model, childIndex, worldMatrix, lightTemplates, outLights);
    }
}

static void LoadLights(const tinygltf::Model& model, SceneBuilder& sceneBuilder)
{
    const std::unordered_map<std::string, LightType> lightTypeMap = {
        {"point", Point},
        {"directional", Directional},
        {"spot", Spot}
    };

    std::vector<LightData> lightTemplates;
    for (const auto& gltf_light : model.lights)
    {
        LightData lightData;
        lightData.type = lightTypeMap.at(gltf_light.type);
        lightData.position = {0, 0, 0};
        lightData.direction = {0, 0, -1};
        lightData.color = { static_cast<float>(gltf_light.color[0]), static_cast<float>(gltf_light.color[1]), static_cast<float>(gltf_light.color[2]) };
        lightData.intensity = static_cast<float>(gltf_light.intensity);
        lightData.range = static_cast<float>(gltf_light.range);

        if (lightData.type == Spot)
        {
            // TODO: Implement inner / outer cone angles.
        }

        lightTemplates.emplace_back(lightData);
    }

    std::vector<LightData> lightDataVector;

    int defaultScene = model.defaultScene >= 0 ? model.defaultScene : 0;
    const auto& sceneRoots = model.scenes[defaultScene].nodes;
    for (int rootIndex : sceneRoots)
    {
        CollectLightsFromNode(model, rootIndex, DirectX::XMMatrixIdentity(), lightTemplates, lightDataVector);
    }

    if (lightDataVector.empty())
    {
        LightData lightData;
        lightData.type = Directional;
        lightData.position = {0, 0, 0};
        lightData.direction = {-0.5f, -0.7071f, -0.067f};
        lightData.color = {1, 1, 1};
        lightData.intensity = 3.0f;
        lightData.range = 0.0f;
        sceneBuilder.AddLightData(lightData);
        return;
    }

    for (const auto& light : lightDataVector)
    {
        sceneBuilder.AddLightData(light);
    }
}

std::shared_ptr<Scene> ModelLoading::LoadScene(Renderer& renderer, const AssetId& assetId)
{
    spdlog::info("Loading scene {}", assetId.AsString());

    std::string extension = assetId.AsPath().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == ".json")
        return TungstenLoading::LoadScene(renderer, assetId);

    tinygltf::Model model;
    bool succeeded = LoadTinyGLTFModel(assetId.AsPath(), model);

    assert(succeeded && "Failed to load model");
    assert(model.scenes.size() > 0 && "Model has no scenes!");
    
    auto scene_builder = SceneBuilder();

    scene_builder.SetName(assetId.AsString());

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unordered_map<int, std::shared_ptr<Texture>> textureCache;
    
    for (auto gltf_model : model.meshes)
    {
        auto current_model = std::make_shared<Model>();

        for (auto& primitive : gltf_model.primitives)
        {
            auto prim = LoadPrimitive(renderer, model, primitive, vertices, indices, textureCache);
            current_model->AddMesh(prim);
        }

        scene_builder.AddModel(current_model);
    }

    LoadLights(model, scene_builder);
    
    auto vertex_index_pair = renderer.CreateSceneResources(vertices, indices);
    auto vertex_buffer = vertex_index_pair.first;
    auto index_buffer = vertex_index_pair.second;
    scene_builder.SetVertexBuffer(vertex_buffer);
    scene_builder.SetIndexBuffer(index_buffer);

    for (auto gltf_model : scene_builder.GetModels())
    {
        for (auto prim : gltf_model->GetMeshes())
        {
            prim->m_vertexBufferOffset.buffer = std::static_pointer_cast<Buffer>(vertex_buffer);
            prim->m_indexBufferOffset.buffer = std::static_pointer_cast<Buffer>(index_buffer);
        }
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
    scene_builder.SetAccelerationStructures(MeshUtils::BuildAccelerationStructures(renderer, scene_builder));
    
    return scene_builder.Build(renderer);
} 