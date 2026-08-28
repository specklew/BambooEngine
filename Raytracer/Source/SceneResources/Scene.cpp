#include "pch.h"
#include "SceneResources/Scene.h"

#include <cfloat>

#include "SceneResources/GameObject.h"
#include "Model.h"
#include "Renderer.h"
#include "Resources/IndexBuffer.h"
#include "Resources/StructuredBuffer.h"
#include "Resources/VertexBuffer.h"
#include "SceneResources/Model.h"
#include "SceneResources/Material.h"
#include "SceneResources/Primitive.h"
#include "SceneResources/SceneNode.h"

SceneBuilder::SceneBuilder()
{
    m_root = std::make_shared<SceneNode>();
}

void SceneBuilder::AddGameObject(const std::shared_ptr<GameObject>& gameObject, const std::shared_ptr<Model>& model)
{
    gameObject->m_model = model;
    m_gameObjects.push_back(gameObject);
}

void SceneBuilder::AddModel(const std::shared_ptr<Model>& model)
{
    m_models.push_back(model);
}

void SceneBuilder::AddChild(const std::shared_ptr<SceneNode>& parent, const std::shared_ptr<SceneNode>& child)
{
    parent->AddChild(child);
}

void SceneBuilder::AddLightData(const LightData& lightData)
{
    m_lightData.push_back(lightData);
}

void SceneBuilder::SetName(const std::string& name)
{
    m_name = name;
}

void SceneBuilder::SetAccelerationStructures(const std::shared_ptr<AccelerationStructures>& accelerationStructures)
{
    m_rtRepresentation = accelerationStructures;
}

void SceneBuilder::SetVertexBuffer(const std::shared_ptr<VertexBuffer>& vertexBuffer)
{
    m_vertexBuffer = vertexBuffer;
}

void SceneBuilder::SetIndexBuffer(const std::shared_ptr<IndexBuffer>& indexBuffer)
{
    m_indexBuffer = indexBuffer;
}

void SceneBuilder::UpdateMatrices()
{
    UpdateMatricesInNodesRecursively(m_root);
}

static std::shared_ptr<StructuredBuffer<GeometryInfo>> CreateGeometryInfoBuffer(Renderer& renderer, const std::vector<std::shared_ptr<Primitive>>& primitives)
{
    std::vector<GeometryInfo> models_info;
    
    for (auto primitive : primitives)
    {
        GeometryInfo info = {};
        info.vertexOffset = primitive->GetVertexView().offset;
        info.indexOffset = primitive->GetIndexView().offset;
        models_info.push_back(info);
    }

    auto geo_buffer = renderer.CreateStructuredBuffer(models_info);
    
    return geo_buffer;
}

static std::shared_ptr<StructuredBuffer<InstanceInfo>> CreateInstanceInfoBuffer(Renderer& renderer,
    const std::vector<std::shared_ptr<GameObject>>& gameObjects,
    const std::vector<std::shared_ptr<Primitive>>& primitives, std::vector<EmissiveTriangle>& outEmissiveTriangles)
{
    std::vector<InstanceInfo> instances_info;

    for (auto go : gameObjects)
    {
        auto model = go->GetModel();

        for (auto primitive : model->GetMeshes())
        {
            auto it = std::find(primitives.begin(), primitives.end(), primitive);
            assert(it != primitives.end() && "Primitive not found in the list of all primitives when creating instance info buffer.");
            int geometry_id = static_cast<int>(std::distance(primitives.begin(), it));
            int texture_id = -1;
            int normal_texture_id = -1;
            int roughness_texture_id = -1;
            float metallic_factor = 1.0f;
            float roughness_factor = 1.0f;
            DirectX::XMFLOAT4 base_color_factor = { 1.0f, 1.0f, 1.0f, 1.0f };
            DirectX::XMFLOAT3 emissive_radiance = { 0.0f, 0.0f, 0.0f };
            DirectX::XMFLOAT3 emissive_texture_average = { 1.0f, 1.0f, 1.0f };
            int emissive_texture_id = -1;

            if (primitive->m_material)
            {
                if (primitive->m_material->m_albedoTexture)
                    texture_id = primitive->m_material->m_albedoTexture->GetTextureIndex();

                if (primitive->m_material->m_normalTexture)
                    normal_texture_id = primitive->m_material->m_normalTexture->GetTextureIndex();

                if (primitive->m_material->m_metallicRoughnessTexture)
                    roughness_texture_id = primitive->m_material->m_metallicRoughnessTexture->GetTextureIndex();

                metallic_factor = primitive->m_material->m_data.metallicFactor;
                roughness_factor = primitive->m_material->m_data.roughnessFactor;
                base_color_factor = primitive->m_material->m_data.baseColorFactor;
                emissive_radiance = primitive->m_material->m_emissiveRadiance;
                emissive_texture_average = primitive->m_material->m_emissiveTextureAverage;

                if (primitive->m_material->m_emissiveTexture)
                    emissive_texture_id = primitive->m_material->m_emissiveTexture->GetTextureIndex();
            }

            InstanceInfo info = {};
            info.geometryId = geometry_id;
            info.textureId = texture_id;
            info.normalTextureId = normal_texture_id;
            info.roughnessTextureId = roughness_texture_id;
            info.metallicFactor = metallic_factor;
            info.roughnessFactor = roughness_factor;
            info.baseColorFactor = base_color_factor;
            info.emissiveRadiance = emissive_radiance;
            info.emissiveLightOffset = -1;
            info.emissiveTextureId = emissive_texture_id;
            // Explicit transpose into the DXR ObjectToWorld3x4 layout.
            const DirectX::XMFLOAT4X4 world = go->GetWorldFloat4X4();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    info.objectToWorld.m[r][c] = world.m[c][r];

            const bool is_emissive = emissive_radiance.x != 0.0f || emissive_radiance.y != 0.0f || emissive_radiance.z != 0.0f;
            if (is_emissive)
            {
                if (primitive->m_emissiveBakePositions.empty() || primitive->m_emissiveBakeIndices.empty()
                    || primitive->m_emissiveBakeUvs.size() != primitive->m_emissiveBakePositions.size())
                {
                    spdlog::error("Material claims emissive radiance but primitive has no baked emissive "
                        "positions/uvs/indices (loader bug); treating instance as non-emissive. instance={} "
                        "geometry_id={}", instances_info.size(), geometry_id);
                }
                else
                {
                    const uint32_t instance_id = static_cast<uint32_t>(instances_info.size());
                    // Emissive-first pool layout (Step 3b): offset is the triangle
                    // index alone, independent of analytic light count, so this
                    // stays valid across runtime light-data rebuilds.
                    info.emissiveLightOffset = static_cast<int>(outEmissiveTriangles.size());

                    const DirectX::XMMATRIX world_matrix = go->GetWorldMatrix();
                    const size_t triangle_count = primitive->m_emissiveBakeIndices.size() / 3;
                    DirectX::XMVECTOR area_weighted_centroid = DirectX::XMVectorZero();
                    float emitter_area = 0.0f;
                    for (size_t t = 0; t < triangle_count; ++t)
                    {
                        const uint32_t i0 = primitive->m_emissiveBakeIndices[t * 3 + 0];
                        const uint32_t i1 = primitive->m_emissiveBakeIndices[t * 3 + 1];
                        const uint32_t i2 = primitive->m_emissiveBakeIndices[t * 3 + 2];

                        const DirectX::XMVECTOR world_v0 = DirectX::XMVector3Transform(DirectX::XMLoadFloat3(&primitive->m_emissiveBakePositions[i0]), world_matrix);
                        const DirectX::XMVECTOR world_v1 = DirectX::XMVector3Transform(DirectX::XMLoadFloat3(&primitive->m_emissiveBakePositions[i1]), world_matrix);
                        const DirectX::XMVECTOR world_v2 = DirectX::XMVector3Transform(DirectX::XMLoadFloat3(&primitive->m_emissiveBakePositions[i2]), world_matrix);

                        EmissiveTriangle tri = {};
                        DirectX::XMStoreFloat3(&tri.v0, world_v0);
                        DirectX::XMStoreFloat3(&tri.v1, world_v1);
                        DirectX::XMStoreFloat3(&tri.v2, world_v2);
                        tri.averageRadiance = { emissive_radiance.x * emissive_texture_average.x,
                            emissive_radiance.y * emissive_texture_average.y,
                            emissive_radiance.z * emissive_texture_average.z };
                        tri.uv0 = primitive->m_emissiveBakeUvs[i0];
                        tri.uv1 = primitive->m_emissiveBakeUvs[i1];
                        tri.uv2 = primitive->m_emissiveBakeUvs[i2];

                        const DirectX::XMVECTOR cross_vec = DirectX::XMVector3Cross(
                            DirectX::XMVectorSubtract(world_v1, world_v0),
                            DirectX::XMVectorSubtract(world_v2, world_v0));
                        const float area = 0.5f * DirectX::XMVectorGetX(DirectX::XMVector3Length(cross_vec));
                        // Never skip a triangle here: emissiveLightOffset + PrimitiveIndex()
                        // contiguity requires exactly one pool entry per primitive triangle.
                        tri.area = (area < 1e-8f) ? 0.0f : area;
                        tri.instanceId = instance_id;

                        emitter_area += tri.area;
                        area_weighted_centroid = DirectX::XMVectorAdd(area_weighted_centroid,
                            DirectX::XMVectorScale(DirectX::XMVectorScale(
                                DirectX::XMVectorAdd(DirectX::XMVectorAdd(world_v0, world_v1), world_v2), 1.0f / 3.0f), tri.area));

                        outEmissiveTriangles.push_back(tri);
                    }

                    // The one place the emitter's authored colour, its texture and the
                    // radiance the light pool will weight it by are visible together.
                    DirectX::XMFLOAT3 centroid = {};
                    if (emitter_area > 0.0f)
                        DirectX::XMStoreFloat3(&centroid, DirectX::XMVectorScale(area_weighted_centroid, 1.0f / emitter_area));
                    spdlog::debug("Emitter instance {}: {} triangles, area={:.3f}, centroid=({:.2f},{:.2f},{:.2f}), "
                        "factor=({:.3f},{:.3f},{:.3f}), emissive texture={}, mean Le=({:.3f},{:.3f},{:.3f})",
                        instance_id, triangle_count, emitter_area, centroid.x, centroid.y, centroid.z,
                        emissive_radiance.x, emissive_radiance.y, emissive_radiance.z, emissive_texture_id,
                        emissive_radiance.x * emissive_texture_average.x, emissive_radiance.y * emissive_texture_average.y,
                        emissive_radiance.z * emissive_texture_average.z);
                }
            }

            instances_info.push_back(info);
        }
    }

    return renderer.CreateStructuredBuffer(instances_info);
}

static std::shared_ptr<StructuredBuffer<LightData>> CreateLightDataBuffer(Renderer& renderer, const std::vector<LightData>& lightData)
{
    return renderer.CreateStructuredBuffer(lightData);
}

static void ComputeWorldAabb(const std::vector<std::shared_ptr<GameObject>>& gameObjects, DirectX::XMFLOAT3& outMin, DirectX::XMFLOAT3& outMax)
{
    using namespace DirectX;
    XMVECTOR vmin = XMVectorReplicate( FLT_MAX);
    XMVECTOR vmax = XMVectorReplicate(-FLT_MAX);
    bool any_seen = false;

    for (const auto& go : gameObjects)
    {
        const auto model = go->GetModel();
        if (!model) continue;

        const XMMATRIX world = go->GetWorldMatrix();

        for (const auto& primitive : model->GetMeshes())
        {
            const XMFLOAT3 lmin = primitive->m_localAabbMin;
            const XMFLOAT3 lmax = primitive->m_localAabbMax;
            if (lmin.x > lmax.x) continue; // empty primitive

            const XMFLOAT3 corners[8] = {
                { lmin.x, lmin.y, lmin.z }, { lmax.x, lmin.y, lmin.z },
                { lmin.x, lmax.y, lmin.z }, { lmax.x, lmax.y, lmin.z },
                { lmin.x, lmin.y, lmax.z }, { lmax.x, lmin.y, lmax.z },
                { lmin.x, lmax.y, lmax.z }, { lmax.x, lmax.y, lmax.z },
            };
            for (const XMFLOAT3& c : corners)
            {
                XMVECTOR p = XMVector3Transform(XMLoadFloat3(&c), world);
                vmin = XMVectorMin(vmin, p);
                vmax = XMVectorMax(vmax, p);
                any_seen = true;
            }
        }
    }

    if (!any_seen)
    {
        outMin = { -1.0f, -1.0f, -1.0f };
        outMax = {  1.0f,  1.0f,  1.0f };
        return;
    }

    XMStoreFloat3(&outMin, vmin);
    XMStoreFloat3(&outMax, vmax);
}

std::shared_ptr<Scene> SceneBuilder::Build(Renderer& renderer)
{
    assert(!m_isBuilt && "Scene has already been built");
    m_isBuilt = true;
    
    UpdateMatricesInNodesRecursively(m_root);

    auto all_prims = GetAllPrimitives();
    
    auto geo_info_buffer = CreateGeometryInfoBuffer(renderer, all_prims);

    std::vector<EmissiveTriangle> emissive_triangles;
    auto instance_info_buffer = CreateInstanceInfoBuffer(renderer, m_gameObjects, all_prims, emissive_triangles);
    auto light_data_buffer = CreateLightDataBuffer(renderer, m_lightData);

    // Emissive-first layout (Step 3b): emissive triangles occupy pool indices
    // [0, emissive_triangles.size()), independent of analytic light count, so
    // InstanceInfo.emissiveLightOffset (baked above) never needs to change when
    // analytic lights are added/removed/edited at runtime. Analytic entries fill
    // the tail and are the only ones a runtime rebuild ever touches.
    std::vector<LightPoolEntry> light_pool;
    light_pool.reserve(emissive_triangles.size() + m_lightData.size());
    constexpr float kPi = 3.14159265358979323846f;
    for (uint32_t i = 0; i < emissive_triangles.size(); ++i)
    {
        const EmissiveTriangle& t = emissive_triangles[i];
        light_pool.push_back({ LightPoolEmissiveTriangle, i, LightPoolLuminance(t.averageRadiance) * t.area * kPi, 0.0f });
    }
    for (uint32_t i = 0; i < m_lightData.size(); ++i)
    {
        light_pool.push_back({ LightPoolAnalytic, i, AnalyticLightPoolPower(m_lightData[i]), 0.0f });
    }
    float light_pool_total_power = 0.0f;
    for (const LightPoolEntry& e : light_pool)
        light_pool_total_power += e.power;
    float running_power = 0.0f;
    for (LightPoolEntry& e : light_pool)
    {
        running_power += e.power;
        e.cdf = (light_pool_total_power > 0.0f) ? running_power / light_pool_total_power : 1.0f;
    }
    if (!light_pool.empty())
        light_pool.back().cdf = 1.0f;

    auto emissive_triangle_buffer = renderer.CreateStructuredBuffer(emissive_triangles);
    auto light_pool_buffer = renderer.CreateStructuredBuffer(light_pool);

    spdlog::info("Light pool built: {} analytic lights, {} emissive triangles, total power={:.3f}",
        m_lightData.size(), emissive_triangles.size(), light_pool_total_power);

    Scene scene;
    scene.m_lightDataCPU = m_lightData;
    ComputeWorldAabb(m_gameObjects, scene.m_aabbMin, scene.m_aabbMax);
    scene.m_gameObjects = std::move(m_gameObjects);
    scene.m_models = std::move(m_models);
    scene.m_root = std::move(m_root);
    scene.m_name = std::move(m_name);
    scene.m_rtRepresentation = std::move(m_rtRepresentation);
    scene.m_vertexBuffer = std::move(m_vertexBuffer);
    scene.m_indexBuffer = std::move(m_indexBuffer);
    scene.m_geometryInfoBuffer = std::move(geo_info_buffer);
    scene.m_instanceInfoBuffer = std::move(instance_info_buffer);
    scene.m_lightDataBuffer = std::move(light_data_buffer);
    scene.m_emissiveTriangleBuffer = std::move(emissive_triangle_buffer);
    scene.m_lightPoolBuffer = std::move(light_pool_buffer);
    scene.m_lightPoolTotalPower = light_pool_total_power;
    scene.m_lightPoolCount = static_cast<uint32_t>(light_pool.size());
    scene.m_emissiveTrianglePower.reserve(emissive_triangles.size());
    for (size_t i = 0; i < emissive_triangles.size(); ++i)
        scene.m_emissiveTrianglePower.push_back(light_pool[i].power);
    scene.m_lightPoolCPU = std::move(light_pool);

    spdlog::info("Scene AABB: min=({:.3f},{:.3f},{:.3f}) max=({:.3f},{:.3f},{:.3f})",
        scene.m_aabbMin.x, scene.m_aabbMin.y, scene.m_aabbMin.z,
        scene.m_aabbMax.x, scene.m_aabbMax.y, scene.m_aabbMax.z);

    return std::make_shared<Scene>(std::move(scene));
}

void Scene::RebuildLightPoolAnalyticTail(Renderer& renderer)
{
    // Emissive prefix is static geometry — its length never changes, so it's
    // read straight from the GPU buffer's fixed element count.
    const size_t emissiveCount = m_emissiveTriangleBuffer ? m_emissiveTriangleBuffer->GetElementsCount() : 0;
    m_lightPoolCPU.resize(emissiveCount);
    // The prefix survives the resize, so only the switched power needs restating.
    for (size_t i = 0; i < emissiveCount && i < m_emissiveTrianglePower.size(); ++i)
        m_lightPoolCPU[i].power = m_emissiveGeometryEnabled ? m_emissiveTrianglePower[i] : 0.0f;
    for (uint32_t i = 0; i < m_lightDataCPU.size(); ++i)
        m_lightPoolCPU.push_back({ LightPoolAnalytic, i, AnalyticLightPoolPower(m_lightDataCPU[i]), 0.0f });

    float totalPower = 0.0f;
    for (const LightPoolEntry& e : m_lightPoolCPU)
        totalPower += e.power;
    float runningPower = 0.0f;
    for (LightPoolEntry& e : m_lightPoolCPU)
    {
        runningPower += e.power;
        e.cdf = (totalPower > 0.0f) ? runningPower / totalPower : 1.0f;
    }
    if (!m_lightPoolCPU.empty())
        m_lightPoolCPU.back().cdf = 1.0f;

    m_lightPoolBuffer = renderer.CreateStructuredBuffer(m_lightPoolCPU);
    m_lightPoolTotalPower = totalPower;
    m_lightPoolCount = static_cast<uint32_t>(m_lightPoolCPU.size());

    spdlog::info("Light pool rebuilt: {} analytic lights, {} emissive triangles ({}), total power={:.3f}",
        m_lightDataCPU.size(), emissiveCount, m_emissiveGeometryEnabled ? "emitting" : "switched off", totalPower);
}

void Scene::SetEmissiveGeometryEnabled(Renderer& renderer, bool enabled)
{
    if (m_emissiveGeometryEnabled == enabled)
        return;

    m_emissiveGeometryEnabled = enabled;
    RebuildLightPoolAnalyticTail(renderer);

    if (!enabled && m_lightPoolTotalPower <= 0.0f)
        spdlog::warn("Emissive geometry is off and no analytic light is set, so the scene has no light "
                     "at all — every indirect path will come back black");
}

void SceneBuilder::UpdateMatricesInNodesRecursively(const std::shared_ptr<SceneNode>& node)
{
    node->UpdateModelConstantBuffer();
    for (const auto& child : node->GetChildren())
    {
        UpdateMatricesInNodesRecursively(child);
    }
}

std::vector<std::shared_ptr<Primitive>> SceneBuilder::GetAllPrimitives() const
{
    std::vector<std::shared_ptr<Primitive>> primitives;

    for (auto model : m_models)
    {
        for (auto primitive : model->GetMeshes())
        {
            primitives.emplace_back(primitive);
        }
    }

    return primitives;
}
