#ifndef TRIANGLE_CLIP_HLSL
#define TRIANGLE_CLIP_HLSL

// Sutherland-Hodgman polygon clipping against an axis-aligned box, ported from
// SIByL vxguiding/include/triangle_clip.hlsli. Used by the geometry bake to
// shrink a triangle to the sliver inside one voxel before taking its AABB
// (voxel.bake.clipping). A triangle clipped by 6 planes has at most 9 vertices.

#define PLANE_THICKNESS_EPSILON 0.00001f

// Classify vertex against an axis-aligned plane: 1 = in front, -1 = behind,
// 0 = on the plane (within thickness epsilon).
int ClassifyAgainstPlane(int sign, int axis, float3 planeVertex, float3 vertex)
{
    const float d = sign * (vertex[axis] - planeVertex[axis]);
    if (d > PLANE_THICKNESS_EPSILON) return 1;
    if (d < -PLANE_THICKNESS_EPSILON) return -1;
    return 0;
}

void ClipPolygonAgainstPlane(
    inout float3 vertices[9],
    inout int vertexCount,
    int sign,
    int axis,
    float3 planeVertex)
{
    int count = vertexCount;
    if (count <= 1) return;

    float3 clipped[9];
    int k = 0;
    bool allOnPlane = true;

    float3 previous = vertices[count - 1];
    int previousSide = ClassifyAgainstPlane(sign, axis, planeVertex, previous);
    for (int j = 0; j < count; ++j)
    {
        float3 current = vertices[j];
        int currentSide = ClassifyAgainstPlane(sign, axis, planeVertex, current);
        if (currentSide == 0)
        {
            if (previousSide != 0)
                clipped[k++] = current;
        }
        else
        {
            allOnPlane = false;
            if (previousSide == 0)
            {
                if (k == 0 || any(clipped[k - 1] != previous))
                    clipped[k++] = previous;
            }
            else if ((currentSide < 0 && previousSide > 0) || (currentSide > 0 && previousSide < 0))
            {
                const float alpha = (current[axis] - planeVertex[axis]) / (current[axis] - previous[axis]);
                clipped[k++] = lerp(current, previous, alpha);
            }

            if (currentSide > 0)
                clipped[k++] = current;
        }

        previous = current;
        previousSide = currentSide;
    }

    if (allOnPlane) return;
    vertexCount = k;
    for (int j2 = 0; j2 < k; ++j2)
        vertices[j2] = clipped[j2];
}

void ClipTriangleAgainstAABB(
    inout float3 vertices[9],
    inout int vertexCount,
    float3 boxMin,
    float3 boxMax)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        ClipPolygonAgainstPlane(vertices, vertexCount, 1, axis, boxMin);
        ClipPolygonAgainstPlane(vertices, vertexCount, -1, axis, boxMax);
    }
}

#undef PLANE_THICKNESS_EPSILON
#endif
