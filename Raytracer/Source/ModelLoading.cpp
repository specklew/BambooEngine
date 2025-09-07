#include "pch.h"
#include "ModelLoading.h"

#include <filesystem>
#include <tinygltf/tiny_gltf.h>

#include "InputElements.h"
#include "Primitive.h"
#include "Renderer.h"
#include "ResourceManager/ResourceManager.h"
#include "ResourceManager/ResourceManagerTypes.h"

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
        vertex.Pos.y = positions[i * 3 + 1] * -1.0f;
        vertex.Pos.z = positions[i * 3 + 2] * -1.0f;

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

    //SPDLOG_DEBUG("Loading glTF model from path: {}", path.string());
    
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
        //SPDLOG_WARN("TinyGLTF warning: {}", warn);
    }
    if (!err.empty()) {
        //SPDLOG_ERROR("TinyGLTF error: {}", err);
    }
    if (!success) {
        //SPDLOG_ERROR("Failed to load glTF model: {}", path);
        return false;
    }

    return true;
}

static void CreateAndPopulateBuffers(Renderer& renderer, Primitive& primitive, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    const uint32_t vbByteSize = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
    const uint32_t ibByteSize = static_cast<uint32_t>(indices.size() * sizeof(uint32_t));

    primitive.vertexBufferCpu = static_cast<BYTE*>(malloc(vbByteSize));
    primitive.indexBufferCpu = static_cast<BYTE*>(malloc(ibByteSize));
    memcpy(primitive.vertexBufferCpu, vertices.data(), vbByteSize);
    memcpy(primitive.indexBufferCpu, indices.data(), ibByteSize);
    
    primitive.vertexByteStride = sizeof(Vertex);
    primitive.vertexBufferByteSize = vbByteSize;
    primitive.indexFormat = DXGI_FORMAT_R32_UINT;
    primitive.indexBufferByteSize = ibByteSize;

    renderer.CreateGpuResourcesForPrimitive(primitive);
}

Primitive ModelLoading::LoadModel(Renderer& renderer, const AssetId& assetId)
{
    tinygltf::Model model;
    bool succeeded = LoadTinyGLTFModel(assetId.AsPath(), model);

    assert(succeeded && "Failed to load model");

    // TODO: Support more.
    if (model.meshes.size() != 1)
    {
        SPDLOG_ERROR("The engine is currently limited to loading models with exactly one mesh. The model at path {} has {} meshes.", assetId.AsString(), model.meshes.size());
        assert(false && "Failed to load model. Only one mesh per model is supported.");
    }

    if (model.meshes[0].primitives.size() != 1)
    {
        SPDLOG_ERROR("The engine is currently limited to loading models with exactly one primitive per mesh. The model at path {} has {} primitives in its first mesh.", assetId.AsString(), model.meshes[0].primitives.size());
        assert(false && "Failed to load model. Only one primitive per mesh is supported.");
    }

    std::vector<Vertex> vertices = {};
    std::vector<uint32_t> indices = {};

    ExtractVertices(model, vertices);
    ExtractIndices(model, indices);
    
    assert(vertices.size() < INT_MAX);
    assert(indices.size() < INT_MAX);

    Primitive primitive(ResourceManager::GetResourceId(assetId));
    primitive.indexCount = indices.size();
    primitive.vertexCount = vertices.size();
    primitive.firstIndex = 0;
    primitive.firstVertex = 0;

    // Here we would normally calculate tangents and AABB.

    CreateAndPopulateBuffers(renderer, primitive, vertices, indices);

    return primitive;
}
