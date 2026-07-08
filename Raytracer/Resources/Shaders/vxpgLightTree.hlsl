// VXPG bottom light tree — a Karras LBVH over the lit voxels, ported from SIByL
// vxguiding/tree/{tree-encode,tree-initial,tree-internal,tree-merge}-pass.slang.
// Runs each frame after clustering; between encode and initialize the leaf codes
// are bitonic-sorted (vxpgBitonicSort.hlsl) so each cluster is a contiguous
// Morton-ordered leaf run, which lets merge derive gClusterRootNodes[32].
//
// Kernel order per frame:
//   ClearCompactToLeaf     -> compact->leaf map reset to -1 (reverse-pdf sentinel)
//   EncodeTreeLeaves       -> per-leaf sort key + the dispatch-args / overflow flag
//   (bitonic sort of the keys)
//   InitializeTreeNodes    -> 2N-1 node array init; leaves get AABB / intensity /
//                             cluster flag / compact->leaf entry
//   BuildTreeInternalNodes -> Karras hierarchy (child + parent links)
//   MergeTreeNodes         -> bottom-up AABB + intensity + cluster-root detection
//
// Three SIByL bugs are deviated on (ADR 0003): args written before the early-out
// (N=0 ghost tree), leaf count clamped to the uint16 node ceiling (+overflow
// flag), and the sort reads the clamped count. Ported from SIByL; identifiers
// renamed (original SIByL names in comments).

#include "LightTreeNode.hlsl"

// uint16 node-index ceiling: the node array holds 2N-1 entries and every index
// must fit uint16, so 2N-1 <= 65535 => N <= 32768.
#define LIGHT_TREE_MAX_LEAVES 32768
// Compact voxel capacity (matches Constants::Graphics::VOXEL_GUIDING_CAPACITY).
#define LIGHT_TREE_COMPACT_CAPACITY 131072
// Sort-key buffer capacity (SIByL bitonic element_count = 65536).
#define LIGHT_TREE_SORT_CAPACITY 65536

cbuffer LightTreeGridCB : register(b0)
{
    float3 gGridMin;
    float gVoxelSize;
    float3 gGridMax;
    uint gGridDim;
}

RWStructuredBuffer<uint64_t> gSortKeys : register(u0); // SIByL u_Codes
// globallycoherent: merge threads communicate through these across thread
// groups (child results read by the sibling's thread). SIByL's Vulkan backend
// is coherent by default; D3D12 UAVs are not without this keyword.
globallycoherent RWStructuredBuffer<LightTreeNode> gNodes : register(u1); // SIByL u_Nodes
RWStructuredBuffer<uint> gLeafRanges : register(u2); // SIByL u_Descendant (dead output; x | y<<16)
RWStructuredBuffer<int> gCompactToLeaf : register(u3); // SIByL compact2leaf
globallycoherent RWStructuredBuffer<int> gClusterRoots : register(u4); // SIByL cluster_roots
RWStructuredBuffer<TreeBuildDispatchArgs> gDispatchArgs : register(u5); // SIByL u_ConstrIndirectArgs
RWStructuredBuffer<uint> gCompactIds : register(u6); // SIByL u_compactIndex (compactID -> voxelID)
RWStructuredBuffer<int> gClusterAssignments : register(u7); // SIByL u_clusterIndex
RWStructuredBuffer<float> gPremulIrradiance : register(u8); // SIByL u_vxIrradiance
RWStructuredBuffer<uint> gVoxCounters : register(u9); // SIByL u_vxCounter ([0] = lit voxel count)
// DEVIATION from SIByL: the merge sibling-gate flag lives in its OWN scalar
// buffer, not TreeNode.flag. DXC will not emit an atomic on a struct sub-member
// (InterlockedCompareExchange on gNodes[i].flag silently compiles NON-atomic ->
// race -> merge parent-walk cycles -> GPU hang). node.flag stays pure clusterID.
globallycoherent RWStructuredBuffer<uint> gNodeVisited : register(u10);

// ---- voxel + Morton helpers (SIByL vxgi_interface / space_filling_curve) ----

int3 ReconstructVoxelCoord(uint voxelID) // SIByL ReconstructIndex
{
    return int3(voxelID % gGridDim,
                (voxelID / gGridDim) % gGridDim,
                voxelID / (gGridDim * gGridDim));
}

void VoxelAABB(int3 coord, out float3 boundMin, out float3 boundMax)
{
    boundMin = gGridMin + float3(coord) * gVoxelSize;
    boundMax = boundMin + gVoxelSize;
}

// Inserts two 0-bits between each of the low 10 bits. SIByL IntegerExplode2Bit.
uint IntegerExplode2Bit(uint x)
{
    x = (x * 0x00010001u) & 0xFF0000FFu;
    x = (x * 0x00000101u) & 0x0F00F00Fu;
    x = (x * 0x00000011u) & 0xC30C30C3u;
    x = (x * 0x00000005u) & 0x49249249u;
    return x;
}

// 30-bit Z-order code for a point in the unit cube. SIByL ZCurve3DToMortonCode.
uint MortonCode3D(float3 unipos)
{
    const float x = min(max(unipos.x * 1024.0f, 0.0f), 1023.0f);
    const float y = min(max(unipos.y * 1024.0f, 0.0f), 1023.0f);
    const float z = min(max(unipos.z * 1024.0f, 0.0f), 1023.0f);
    const uint xx = IntegerExplode2Bit(uint(x));
    const uint yy = IntegerExplode2Bit(uint(y));
    const uint zz = IntegerExplode2Bit(uint(z));
    return xx * 4u + yy * 2u + zz;
}

// ---- ClearCompactToLeaf ----------------------------------------------------
// Bamboo additions cleared per frame:
//  - compact->leaf map -> -1 so an out-of-tree (overflow / unlit) voxel returns
//    "no leaf" in the reverse pdf query instead of a stale leaf index.
//  - the ENTIRE sort-key buffer -> NULL (max). The encode kernel only writes the
//    first numVXs keys; the tail must be max so the fixed worst-case 65536 sort
//    network (which over-dispatches and reads Index1 unguarded) sees padding, not
//    stale garbage that would swap down into the valid region and corrupt the
//    sort (-> a cyclic Karras tree -> merge parent-walk hang -> device removed).

[numthreads(256, 1, 1)]
void ClearCompactToLeaf(uint3 dtid : SV_DispatchThreadID)
{
    // ~uint64_t(0), NOT a hex literal: DXC silently truncates the "uL" suffix
    // to 32 bits (0xFFFFFFFFFFFFFFFFuL compiles to i64 4294967295), which made
    // the padding sort BELOW real keys and corrupt the tree. Proven via -Fc.
    if (dtid.x < LIGHT_TREE_SORT_CAPACITY)
        gSortKeys[dtid.x] = ~uint64_t(0);
    if (dtid.x < LIGHT_TREE_COMPACT_CAPACITY)
        gCompactToLeaf[dtid.x] = -1;
}

// ---- EncodeTreeLeaves (tree-encode-pass) -----------------------------------

[numthreads(256, 1, 1)]
void EncodeTreeLeaves(uint3 dtid : SV_DispatchThreadID)
{
    const int rawCount = int(gVoxCounters[0]);
    // Bug 2 fix: clamp to the uint16 node ceiling so child/parent indices never
    // wrap (a wrap aliases nodes and makes merge's parent-walk cycle -> TDR).
    const int numVXs = min(rawCount, LIGHT_TREE_MAX_LEAVES);
    const int tid = int(dtid.x);

    // Bug 1 fix: write the dispatch args BEFORE the early-out. SIByL wrote them
    // below `if (tid >= numVXs) return;`, so an all-dark frame (numVXs == 0)
    // left the args stale and downstream kernels built a ghost tree.
    if (tid == 0)
    {
        const int numInternal = max(numVXs - 1, 0);
        const int numTotal = max(numVXs * 2 - 1, 0);
        TreeBuildDispatchArgs args;
        args.dispatchLeaf = int3((numVXs + 255) / 256, 1, 1);
        args.numValidVoxels = uint(numVXs);
        args.dispatchInternal = int3((numInternal + 255) / 256, 1, 1);
        args.overflowFlag = (rawCount > LIGHT_TREE_MAX_LEAVES) ? 1u : 0u;
        args.dispatchNode = int3((numTotal + 255) / 256, 1, 1);
        args.padding1 = 0u;
        args.drawRects = int4(6, numTotal, 0, 0);
        gDispatchArgs[0] = args;
    }

    if (tid >= numVXs)
        return;

    const uint leafID = uint(tid);
    const uint voxelID = gCompactIds[leafID];
    const int3 mapPos = ReconstructVoxelCoord(voxelID);
    const uint clusterID = uint(gClusterAssignments[leafID]);

    const float3 unipos = clamp(float3(mapPos) + 0.5f, 0.0f, float(gGridDim)) / float(gGridDim);
    const uint64_t posCode = uint64_t(MortonCode3D(unipos));
    const uint64_t clusterCode = uint64_t(clusterID);
    const uint64_t idCode = uint64_t(leafID);
    gSortKeys[leafID] = (clusterCode << 48) | (posCode << 16) | (idCode << 0);
}

// ---- InitializeTreeNodes (tree-initial-pass) -------------------------------

[numthreads(256, 1, 1)]
void InitializeTreeNodes(uint3 dtid : SV_DispatchThreadID)
{
    const int numVXs = int(gDispatchArgs[0].numValidVoxels);
    const int tid = int(dtid.x);
    const int numInternal = numVXs - 1;
    const int numTotal = numVXs * 2 - 1;

    if (tid < 32)
        gClusterRoots[tid] = -1;

    if (tid >= numTotal)
        return;

    gNodeVisited[tid] = 0u; // merge sibling-gate reset (was TreeNode.flag in SIByL)

    LightTreeNode node;
    node.parentIndex = 0xFFFF;
    node.leftIndex = 0xFFFF;
    node.rightIndex = 0xFFFF;
    node.voxelIndex = 0xFFFF;
    node.flag = 0u;
    node.intensity = 0.0f;
    node.aabbMin = uint2(0u, 0u);
    node.aabbMax = uint2(0u, 0u);

    if (tid < numInternal)
    {
        gNodes[tid] = node;
        return;
    }

    const uint leafID = uint(tid - numInternal);
    const uint compactID = uint(gSortKeys[leafID] & 0xFFFFuL);
    const uint voxelID = gCompactIds[compactID];
    const int3 mapPos = ReconstructVoxelCoord(voxelID);
    const uint clusterID = uint(gClusterAssignments[compactID]);

    float3 boundMin, boundMax;
    VoxelAABB(mapPos, boundMin, boundMax);
    node.aabbMin = PackFloat3(boundMin);
    node.aabbMax = PackFloat3(boundMax);
    node.intensity = gPremulIrradiance[compactID];
    node.voxelIndex = uint16_t(compactID);
    node.flag = clusterID;

    gLeafRanges[tid] = leafID; // dead output (cost fidelity)
    gCompactToLeaf[compactID] = tid;
    gNodes[tid] = node;
}

// ---- BuildTreeInternalNodes (tree-internal-pass) ---------------------------

int Clz(uint x)
{
    return 31 - int(firstbithigh(x));
}

int Clz64(uint64_t x)
{
    const int h = Clz(uint(x >> 32));
    const int l = Clz(uint(x & 0xFFFFFFFFuL));
    return h == 32 ? h + l : h;
}

int CommonUpperBits(uint64_t lhs, uint64_t rhs)
{
    return Clz64(lhs ^ rhs);
}

void SwapUint(inout uint u1, inout uint u2)
{
    uint tmp = u1;
    u1 = u2;
    u2 = tmp;
}

uint2 DetermineRange(uint numLeaves, uint idx)
{
    if (idx == 0)
        return uint2(0, numLeaves - 1);

    const uint64_t selfCode = gSortKeys[idx];
    const int lDelta = CommonUpperBits(selfCode, gSortKeys[idx - 1]);
    const int rDelta = CommonUpperBits(selfCode, gSortKeys[idx + 1]);
    const int d = (rDelta > lDelta) ? 1 : -1;

    const int deltaMin = min(lDelta, rDelta);
    int lMax = 2;
    int delta = -1;
    int iTmp = int(idx) + d * lMax;
    if (0 <= iTmp && iTmp < int(numLeaves))
        delta = CommonUpperBits(selfCode, gSortKeys[iTmp]);
    while (delta > deltaMin)
    {
        lMax <<= 1;
        iTmp = int(idx) + d * lMax;
        delta = -1;
        if (0 <= iTmp && iTmp < int(numLeaves))
            delta = CommonUpperBits(selfCode, gSortKeys[iTmp]);
    }

    int l = 0;
    int t = lMax >> 1;
    while (t > 0)
    {
        iTmp = int(idx) + (l + t) * d;
        delta = -1;
        if (0 <= iTmp && iTmp < int(numLeaves))
            delta = CommonUpperBits(selfCode, gSortKeys[iTmp]);
        if (delta > deltaMin)
            l += t;
        t >>= 1;
    }

    uint jdx = idx + uint(l * d);
    if (d < 0)
        SwapUint(idx, jdx);
    return uint2(idx, jdx);
}

uint FindSplit(uint first, uint last)
{
    const uint64_t firstCode = gSortKeys[first];
    const uint64_t lastCode = gSortKeys[last];
    if (firstCode == lastCode)
        return (first + last) >> 1;

    const int deltaNode = CommonUpperBits(firstCode, lastCode);
    int split = int(first);
    int stride = int(last - first);
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < int(last))
        {
            const int delta = CommonUpperBits(firstCode, gSortKeys[middle]);
            if (delta > deltaNode)
                split = middle;
        }
    } while (stride > 1);

    return uint(split);
}

[numthreads(256, 1, 1)]
void BuildTreeInternalNodes(uint3 dtid : SV_DispatchThreadID)
{
    const uint numVPLs = gDispatchArgs[0].numValidVoxels;
    const int idx = int(dtid.x);
    const int numObjects = int(numVPLs);

    if (idx >= max(0, numObjects - 1))
        return;

    const uint2 ij = DetermineRange(numVPLs, uint(idx));
    gLeafRanges[idx] = ij.x | (ij.y << 16); // dead output
    const uint gamma = FindSplit(ij.x, ij.y);

    // Clamped N <= 32768 keeps gamma + (numVPLs-1) <= 65534 < 0xFFFF, so these
    // uint16 adds never wrap (SIByL Bug 2 site, defused by the encode clamp).
    uint16_t leftIndex = uint16_t(gamma);
    uint16_t rightIndex = uint16_t(gamma + 1);
    if (min(ij.x, ij.y) == gamma)
        leftIndex += uint16_t(numVPLs - 1);
    if (max(ij.x, ij.y) == gamma + 1)
        rightIndex += uint16_t(numVPLs - 1);

    gNodes[idx].leftIndex = leftIndex;
    gNodes[idx].rightIndex = rightIndex;
    gNodes[leftIndex].parentIndex = uint16_t(idx);
    gNodes[rightIndex].parentIndex = uint16_t(idx);
}

// ---- MergeTreeNodes (tree-merge-pass) --------------------------------------

[numthreads(256, 1, 1)]
void MergeTreeNodes(uint3 dtid : SV_DispatchThreadID)
{
    const uint numVPLs = gDispatchArgs[0].numValidVoxels;
    if (dtid.x >= numVPLs)
        return;
    const int numInternalNodes = int(numVPLs - 1);
    const int idx = int(dtid.x) + numInternalNodes;

    uint16_t parent = gNodes[idx].parentIndex;

    if (numVPLs == 1 && parent == 0xFFFF)
    {
        gClusterRoots[min(gNodes[idx].flag, 31u)] = 0;
        return;
    }

    // DEVIATION from SIByL: bail before dereferencing an invalid parent. SIByL
    // unconditionally reads u_Nodes[parent].left_idx first, which for
    // parent == 0xFFFF reads one element past the node buffer.
    if (parent == 0xFFFF)
        return;

    int lhsNodeId = gNodes[parent].leftIndex;
    LightTreeNode lhs = gNodes[lhsNodeId];

    while (parent != 0xFFFF)
    {
        uint old;
        InterlockedCompareExchange(gNodeVisited[parent], 0u, 1u, old);
        if (old == 0)
            return; // first thread here waits for the sibling

        const uint lidx = gNodes[parent].leftIndex;
        const uint ridx = gNodes[parent].rightIndex;
        const uint rhsNodeId = (uint(lhsNodeId) != ridx) ? ridx : lidx;
        LightTreeNode rhs = gNodes[rhsNodeId];

        LightTreeNode merged = gNodes[parent];
        merged.intensity = lhs.intensity + rhs.intensity;
        const float3 boundMin = min(UnpackFloat3(lhs.aabbMin), UnpackFloat3(rhs.aabbMin));
        const float3 boundMax = max(UnpackFloat3(lhs.aabbMax), UnpackFloat3(rhs.aabbMax));
        merged.aabbMin = PackFloat3(boundMin);
        merged.aabbMax = PackFloat3(boundMax);

        const bool clusterSplit = lhs.flag != rhs.flag;
        merged.flag = (!clusterSplit) ? lhs.flag : 0xFFFFFFFFu;
        // DEVIATION from SIByL (bug 4): every cluster_roots write is bounded to
        // < 32. SIByL's root-node write has no 0xFFFFFFFF guard, and on any tree
        // whose two root subtrees BOTH span multiple clusters (the normal case)
        // it writes cluster_roots[0xFFFFFFFF] -> wild GPU write -> device removed.
        if (lhs.flag != merged.flag && lhs.flag < 32u)
            gClusterRoots[lhs.flag] = lhsNodeId;
        if (rhs.flag != merged.flag && rhs.flag < 32u)
            gClusterRoots[rhs.flag] = int(rhsNodeId);
        if (merged.parentIndex == 0xFFFF && lhs.flag == rhs.flag && merged.flag < 32u)
            gClusterRoots[merged.flag] = 0;

        gNodes[parent] = merged;

        lhs = merged;
        lhsNodeId = int(parent);
        parent = gNodes[parent].parentIndex;
    }
}
