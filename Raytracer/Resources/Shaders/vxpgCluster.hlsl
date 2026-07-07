// VXPG cluster pass — groups fingerprinted lit voxels into 32 supervoxels
// (the MRCS "column clustering"). Two kernels run each frame after the
// fingerprint pass:
//
//   SeedClusterCenters   (port of mrcs/column-kmpp-seeding.slang)
//     Single 1024-thread group. Every thread drafts one random lit voxel as a
//     candidate; 31 k-means++ rounds then pick seeds far apart (selection
//     probability ~ squared distance to the nearest already-picked center)
//     via a two-level wave reduction: 32 warps x 32 lanes.
//
//   AssignVoxelClusters  (port of mrcs/column-find-center.slang)
//     Every compact voxel compares its fingerprint against the 32 centers and
//     stores the nearest cluster id.
//
// Distance = Hamming(fingerprints) + |premultiplied-irradiance difference|
// (SIByL ComputeDistance with extra=true, position_weight=0, intensity_weight=1
// — the canonical VXPGGraph values; see ADR 0003). Seeding-only k-means++:
// bitmask centroids have no mean, so the seeds ARE the cluster descriptors.
//
// Ported from SIByL; identifiers renamed to descriptive Bamboo names (original
// SIByL names kept in comments for traceability).

#include "Random.hlsl"

#define CLUSTER_COUNT 32

cbuffer ClusterCB : register(b0)
{
    uint gGridDim;
    // 0 = SIByL-faithful frame-constant seeding (its sampler is seeded with
    // hardcoded zeros); nonzero = per-frame hash (vxpg.cluster.frameVaryingSeed).
    uint gClusterSeedFrameTerm;
    uint _clusterPad0;
    uint _clusterPad1;
}

// SIByL svoxel_info (mrcs/cluster-common.hlsli).
struct ClusterCenter
{
    uint4  fingerprint; // SIByL desc_info: the 128-bit visibility signature
    float3 position;    // SIByL center: voxel-coordinate space
    float  intensity;   // premultiplied irradiance
};

RWStructuredBuffer<int>           gClusterSeedCompactIds   : register(u0); // SIByL u_Seeds
RWStructuredBuffer<ClusterCenter> gClusterCenters          : register(u1); // SIByL u_RowClusterInfo / u_ClusterInfo
RWStructuredBuffer<uint4>         gGuidingDispatchArgs     : register(u2); // SIByL u_IndirectArgs ([0].w = lit voxel count)
RWStructuredBuffer<uint4>         gVoxelFingerprints       : register(u3); // SIByL u_RowVisibility (uint4 view of the mask words)
RWStructuredBuffer<uint>          gCompactIds              : register(u4); // SIByL u_CompactIndices (compactID -> voxelID)
RWStructuredBuffer<float>         gPremulIrradiance        : register(u5); // SIByL u_PremulIrradiance
RWStructuredBuffer<int>           gVoxelClusterAssignments : register(u6); // SIByL u_Clusters

// SIByL call-site weights (kept as named constants; position term is dead but
// the struct keeps the field for port fidelity).
static const float POSITION_WEIGHT  = 0.0;
static const float INTENSITY_WEIGHT = 1.0;

float NextRandom(inout uint state)
{
    state = pcg_hash(state);
    return float(state) * (1.0 / 4294967296.0);
}

// Inverts CompactVoxels' flatId = x + y*dim + z*dim^2.
int3 ReconstructVoxelCoord(uint flatId)
{
    return int3(flatId % gGridDim,
                (flatId / gGridDim) % gGridDim,
                flatId / (gGridDim * gGridDim));
}

// SIByL ComputeDistance (mrcs/cluster-common.hlsli), extra = true.
float ClusterDistance(uint4 fingerprintA, uint4 fingerprintB,
                      float3 positionA, float3 positionB,
                      float intensityA, float intensityB)
{
    const uint4 diff = fingerprintA ^ fingerprintB;
    const float hamming = countbits(diff.x) + countbits(diff.y)
                        + countbits(diff.z) + countbits(diff.w);
    return hamming + POSITION_WEIGHT * distance(positionA, positionB)
                   + INTENSITY_WEIGHT * abs(intensityA - intensityB);
}

// ---- SeedClusterCenters ----------------------------------------------------

groupshared float  sWarpProbability[32];       // SIByL warp_prob
groupshared uint4  sCurrentCenterFingerprint;  // SIByL current_center
groupshared float3 sCurrentCenterPosition;     // SIByL current_center_pos
groupshared float  sCurrentCenterIntensity;    // SIByL current_center_intensity
groupshared int    sSelectedWarp;              // SIByL selected_cluster

[numthreads(1024, 1, 1)]
[WaveSize(32)]
void SeedClusterCenters(uint3 tid : SV_DispatchThreadID)
{
    const uint threadId = tid.x;
    const uint laneId = WaveGetLaneIndex();
    const uint warpId = threadId / 32u;
    const int litVoxelCount = int(gGuidingDispatchArgs[0].w);

    uint randState = pcg_hash((threadId * 9781u + gClusterSeedFrameTerm * 26699u) | 1u);

    // max() guards litVoxelCount == 0: SIByL clamps to -1 there and reads
    // u_RowVisibility[-1] — an OOB read through an unchecked root UAV here.
    const int candidateId = clamp(int(NextRandom(randState) * litVoxelCount),
                                  0, max(litVoxelCount - 1, 0));

    const int voxelId = int(gCompactIds[candidateId]);
    const float3 candidatePosition = float3(ReconstructVoxelCoord(uint(voxelId)));
    const uint4 candidateFingerprint = gVoxelFingerprints[candidateId];
    const float candidateIntensity = gPremulIrradiance[candidateId];

    if (threadId == 0u)
    {
        // random choice of the first seed
        sCurrentCenterFingerprint = candidateFingerprint;
        sCurrentCenterPosition = candidatePosition;
        sCurrentCenterIntensity = candidateIntensity;
        gClusterSeedCompactIds[0] = candidateId;
        ClusterCenter first;
        first.fingerprint = candidateFingerprint;
        first.position = candidatePosition;
        first.intensity = candidateIntensity;
        gClusterCenters[0] = first;
    }
    if (warpId == 0u)
        sWarpProbability[laneId] = 0.0;
    GroupMemoryBarrierWithGroupSync();

    float nearestCenterDistance = 100000000.0;

    for (int seedId = 1; seedId < CLUSTER_COUNT; ++seedId)
    {
        // Distance to the newest center only — the running min makes it the
        // nearest over all picked centers. Port-faithful quirk: SIByL refreshes
        // only the center FINGERPRINT after each pick; the position/intensity
        // compared against stay those of seed 0.
        const float d = ClusterDistance(candidateFingerprint, sCurrentCenterFingerprint,
                                        candidatePosition, sCurrentCenterPosition,
                                        candidateIntensity, sCurrentCenterIntensity);
        nearestCenterDistance = min(nearestCenterDistance, d);

        // k-means++: selection probability ~ squared distance
        float weight = nearestCenterDistance * nearestCenterDistance;
        const float warpWeightSum = WaveActiveSum(weight);
        if (WaveIsFirstLane())
            sWarpProbability[warpId] = warpWeightSum;

        GroupMemoryBarrierWithGroupSync();

        // Warp 0 re-purposes its lanes to hold the 32 warp sums; lane 0 zeroes
        // itself in SIByL (excludes warp 0's own sum from the top level).
        if (warpId == 0u)
        {
            weight = sWarpProbability[laneId];
            if (WaveIsFirstLane()) weight = 0.0;
        }

        // Butterfly reduction storing, per level, the probability of taking
        // the "left" child — 5 floats per thread instead of a shared 64-float
        // tree. WaveReadLaneAt with a per-lane index is spec-gray in DXIL but
        // is a real shuffle on the NV/AMD targets (Slang WaveShuffle maps to it).
        float leftProbability[6];
        for (int level = 0; level < 5; ++level)
        {
            const float neighborWeight = WaveReadLaneAt(weight, laneId ^ (1u << level));
            const float weightSum = weight + neighborWeight;
            leftProbability[4 - level] = (weightSum == 0.0) ? 0.5 : weight / weightSum;
            weight = weightSum;
        }

        // Sample the implicit binary tree top-down; warp 0's traversal picks
        // the warp, the picked warp's own traversal picks the lane.
        float rnd = NextRandom(randState);
        int nodeId = 0;
        for (int level = 0; level < 5; ++level)
        {
            const float leftP = leftProbability[level];
            if (rnd < leftP)
            {
                rnd /= leftP;
            }
            else
            {
                nodeId += int(16u >> level);
                rnd = (rnd - leftP) / (1.0 - leftP);
            }
            leftProbability[level + 1] = WaveReadLaneAt(leftProbability[level + 1], nodeId);
        }

        if (threadId == 0u)
            sSelectedWarp = nodeId;

        GroupMemoryBarrierWithGroupSync();

        const int selectedLane = WaveReadLaneAt(nodeId, 0);
        if (int(warpId) == sSelectedWarp && int(laneId) == selectedLane)
        {
            gClusterSeedCompactIds[seedId] = candidateId;
            ClusterCenter center;
            center.fingerprint = candidateFingerprint;
            center.position = candidatePosition;
            center.intensity = candidateIntensity;
            gClusterCenters[seedId] = center;
            sCurrentCenterFingerprint = candidateFingerprint; // SIByL: fingerprint only
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

// ---- AssignVoxelClusters ----------------------------------------------------

[numthreads(256, 1, 1)]
void AssignVoxelClusters(uint3 tid : SV_DispatchThreadID)
{
    const uint litVoxelCount = gGuidingDispatchArgs[0].w;
    const uint compactId = tid.x;
    if (compactId >= litVoxelCount) return; // over-dispatch early-out (option b)

    const int voxelId = int(gCompactIds[compactId]);
    const float3 voxelPosition = float3(ReconstructVoxelCoord(uint(voxelId)));
    const float intensity = gPremulIrradiance[compactId];
    const uint4 fingerprint = gVoxelFingerprints[compactId];

    int nearestCluster = -1;
    float nearestDistance = 999999.9999;
    for (int clusterId = 0; clusterId < CLUSTER_COUNT; ++clusterId)
    {
        const ClusterCenter center = gClusterCenters[clusterId];
        const float d = ClusterDistance(fingerprint, center.fingerprint,
                                        voxelPosition, center.position,
                                        intensity, center.intensity);
        if (d < nearestDistance)
        {
            nearestDistance = d;
            nearestCluster = clusterId;
        }
    }

    gVoxelClusterAssignments[compactId] = nearestCluster;
}
