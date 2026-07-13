#pragma once

#include <vector>
#include <cstdint>
#include <memory>

#include "InputElements.h" // Vertex

class Renderer;
class SceneBuilder;
class AccelerationStructures;

// Shared load-time mesh pipeline (ADR 0005). Both scene loaders (glTF, Tungsten)
// run geometry through these so every vertex reaches the GPU in the canonical
// left-handed space with valid, self-consistent attributes — no per-model repair.
namespace MeshUtils
{
    // glTF/Tungsten spec: primitives without authored normals shade with face
    // normals. Area-weighted per-vertex accumulation, standard cross(v1-v0, v2-v0)
    // of the canonical CCW winding (matches the shader tri_normal).
    void ComputeNormals(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

    // mikktspace tangent generation (reads normals — run after ComputeNormals).
    void ComputeTangents(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

    // Drop only truly degenerate (zero-area / collinear) triangles; thin flat
    // shells keep real area and survive.
    void DropDegenerateTriangles(const std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

    // Guarantee a unit normal and an orthonormal tangent frame per vertex; scrub
    // non-finite positions/uvs; rebuild degenerate tangents from the normal.
    void EnforceVertexInvariants(std::vector<Vertex>& vertices);

    // Build BLAS per primitive + one TLAS over the builder's game objects.
    std::shared_ptr<AccelerationStructures> BuildAccelerationStructures(const Renderer& renderer, const SceneBuilder& scene);
}
