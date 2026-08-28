#ifndef LIGHT_TREE_NODE_HLSL
#define LIGHT_TREE_NODE_HLSL

// Port of SIByL vxguiding/tree/shared.hlsli (TreeNode + TreeConstrIndirectArgs).
// SIByL stores the four indices as uint16_t; here they are uint, which lifts the
// 2N-1 <= 65535 node ceiling that capped the tree at 32768 leaves. The node grows
// 32 -> 40 bytes in exchange. Identifiers renamed to descriptive Bamboo names
// (original SIByL names kept in comments for port traceability).

// Sentinel for "no such node". Was 0xFFFF while the indices were 16-bit.
#define LIGHT_TREE_NO_NODE 0xFFFFFFFFu

struct LightTreeNode // SIByL TreeNode
{
    uint2 aabbMin; // packed float3 (half x/y/z), SIByL aabbMin
    uint2 aabbMax; // SIByL aabbMax
    float intensity; // leaf: PremulIrradiance; internal: child intensity sum
    uint flag; // clusterID; 0xFFFFFFFF on an internal node whose subtree spans
               // several clusters. NOT the merge visited-flag: that lives in its
               // own buffer (gNodeVisited, see vxpgLightTree.hlsl).
    uint parentIndex; // SIByL parent_idx (LIGHT_TREE_NO_NODE = none / root)
    uint leftIndex; // SIByL left_idx
    uint rightIndex; // SIByL right_idx
    uint voxelIndex; // SIByL vx_idx; LIGHT_TREE_NO_NODE for internal nodes, else compactID
};

struct TreeBuildDispatchArgs // SIByL TreeConstrIndirectArgs
{
    int3 dispatchLeaf; // SIByL dispatch_leaf
    uint numValidVoxels; // SIByL numValidVPLs; CLAMPED leaf count. The sort reads
                         // this at byte offset 12 (SIByL counter_offset mechanism).
    int3 dispatchInternal; // SIByL dispatch_internal
    uint overflowFlag; // SIByL padding0 repurposed: 1 = lit voxel count exceeded
                       // the leaf cap this frame (surfaced as debug-view magenta)
    int3 dispatchNode; // SIByL dispatch_node
    uint padding1; // SIByL padding1
    int4 drawRects; // SIByL draw_rects
};

// SIByL packFloat3/unpackFloat3 (packing.hlsli): x/y as an f16 pair in u.x,
// z as an f16 in the high half of u.y. Lossy (half precision) but faithful, and
// f16 error (~0.01 at Sponza scale) is far below one voxel edge.
uint2 PackFloat3(float3 v)
{
    uint2 u;
    u.x = f32tof16(v.x) | (f32tof16(v.y) << 16);
    u.y = f32tof16(v.z) << 16;
    return u;
}

float3 UnpackFloat3(uint2 u)
{
    return float3(f16tof32(u.x & 0xFFFF), f16tof32(u.x >> 16), f16tof32(u.y >> 16));
}

#endif // LIGHT_TREE_NODE_HLSL
