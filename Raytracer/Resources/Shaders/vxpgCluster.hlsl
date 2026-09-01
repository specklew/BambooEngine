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
//   AccumulateClusterCenters / UpdateClusterCenters  (no SIByL counterpart)
//     The Lloyd half of k-means++: sum each cluster's members, replace the
//     center with their centroid, assign again. Four rounds.
//
// Distance = Hamming(fingerprints) + |premultiplied-irradiance difference|
// (SIByL ComputeDistance with extra=true, position_weight=0, intensity_weight=1
// — the canonical VXPGGraph values; see ADR 0003).
//
// SIByL stops after seeding, on the argument that a bitmask has no mean. It has
// one: under Hamming distance the centroid of a set of bit vectors is the
// PER-BIT MAJORITY, which is still a bit vector, so the Lloyd step is well
// defined and does decrease the objective. The intensity term is L1, whose exact
// minimiser is the median rather than the mean; the mean is used because it is
// what k-means prescribes and because a median needs a sort this kernel does not
// have. That approximation is the only inexactness in the update.
//
// Ported from SIByL; identifiers renamed to descriptive Bamboo names (original
// SIByL names kept in comments for traceability).

#include "PassRegisters.h"
#include "Random.hlsl"

#define CLUSTER_COUNT 32

cbuffer ClusterCB : BAMBOO_PASS_CBV(CLUSTER_REG_CB)
{
    uint gGridDim;
    // 0 = SIByL-faithful frame-constant seeding (its sampler is seeded with
    // hardcoded zeros); nonzero = per-frame hash (vxpg.cluster.frameVaryingSeed).
    uint gClusterSeedFrameTerm;
    // One-shot diagnostic (vxpg.cluster.dumpStats). Off, the atomics below are
    // branched out entirely so a benchmark frame never pays for them.
    uint gCollectStats;
    uint _clusterPad1;
}

// Diagnostic layout, 4 x CLUSTER_COUNT uints. Population plus the three numbers
// that separate the possible explanations for a lopsided clustering:
//   HAMMING   — realised bit distance to the assigned center
//   INTENSITY — the other distance term (fixed point x1000); whichever of the two
//               is larger is the one actually choosing clusters
//   POPCOUNT  — how many of the 128 representatives each voxel sees at all. Near
//               0 means the fingerprints are empty (a fingerprint-pass fault),
//               near 128 means everything sees everything, and a mid value with a
//               small Hamming means the visibility really is that uniform — a
//               property of the scene, not a bug.
#define CLUSTER_STAT_POPULATION 0
#define CLUSTER_STAT_HAMMING    CLUSTER_COUNT
#define CLUSTER_STAT_INTENSITY  (CLUSTER_COUNT * 2)
#define CLUSTER_STAT_POPCOUNT   (CLUSTER_COUNT * 3)
#define CLUSTER_STAT_COUNT      (CLUSTER_COUNT * 4)
#define CLUSTER_STAT_INTENSITY_SCALE 1000.0

// Lloyd accumulator, per cluster: one counter per fingerprint bit, then the
// position and intensity sums and the population they are divided by.
#define CLUSTER_ACC_BITS        0
#define CLUSTER_ACC_POSITION    128
#define CLUSTER_ACC_INTENSITY   131
#define CLUSTER_ACC_POPULATION  132
#define CLUSTER_ACC_STRIDE      133
#define CLUSTER_ACC_COUNT       (CLUSTER_COUNT * CLUSTER_ACC_STRIDE)
// The two float terms are summed through integer atomics, so they carry a scale.
// Headroom at 256: the assignment buffer holds at most 131072 voxels, so the sum
// stays under 2^32 up to a mean intensity of 128 — the scenes measure 0.20
// (bistro-exterior) to 11.96 (veach-ajar) with vxpg.cluster.dumpStats, an order
// of magnitude below. Voxel coordinates are bounded by the grid dimension, so
// their scale can be lower and still never wrap.
#define CLUSTER_ACC_INTENSITY_SCALE 256.0
#define CLUSTER_ACC_POSITION_SCALE  16.0
// Lloyd rounds after the seeding. k-means++ seeds are already spread apart, so
// the assignment stops moving after a handful; four costs about 1% of a guided
// frame, which is inside the noise of the measurement it feeds.
#define CLUSTER_LLOYD_ITERATIONS 4

// SIByL svoxel_info (mrcs/cluster-common.hlsli).
struct ClusterCenter
{
    uint4  fingerprint; // SIByL desc_info: the 128-bit visibility signature
    float3 position;    // SIByL center: voxel-coordinate space
    float  intensity;   // premultiplied irradiance
};

RWStructuredBuffer<int>           gClusterSeedCompactIds   : BAMBOO_PASS_UAV(CLUSTER_REG_SEED_COMPACT_IDS); // SIByL u_Seeds
RWStructuredBuffer<ClusterCenter> gClusterCenters          : BAMBOO_PASS_UAV(CLUSTER_REG_CENTERS); // SIByL u_RowClusterInfo / u_ClusterInfo
RWStructuredBuffer<uint4>         gGuidingDispatchArgs     : BAMBOO_PASS_UAV(CLUSTER_REG_DISPATCH_ARGS); // SIByL u_IndirectArgs ([0].w = lit voxel count)
RWStructuredBuffer<uint4>         gVoxelFingerprints       : BAMBOO_PASS_UAV(CLUSTER_REG_FINGERPRINTS); // SIByL u_RowVisibility (uint4 view of the mask words)
RWStructuredBuffer<uint>          gCompactIds              : BAMBOO_PASS_UAV(CLUSTER_REG_COMPACT_IDS); // SIByL u_CompactIndices (compactID -> voxelID)
RWStructuredBuffer<float>         gPremulIrradiance        : BAMBOO_PASS_UAV(CLUSTER_REG_PREMUL_IRRADIANCE); // SIByL u_PremulIrradiance
RWStructuredBuffer<int>           gVoxelClusterAssignments : BAMBOO_PASS_UAV(CLUSTER_REG_ASSIGNMENTS); // SIByL u_Clusters
RWStructuredBuffer<uint>          gClusterStats            : BAMBOO_PASS_UAV(CLUSTER_REG_STATS); // Bamboo diagnostic, no SIByL counterpart
RWStructuredBuffer<uint>          gClusterAccumulators     : BAMBOO_PASS_UAV(CLUSTER_REG_ACCUMULATORS); // Lloyd sums, no SIByL counterpart

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

#define CLUSTER_SEED_THREADS 1024 // numthreads of SeedClusterCenters

// ---- SeedClusterCenters ----------------------------------------------------

groupshared float  sWarpProbability[32];       // SIByL warp_prob
groupshared uint4  sCurrentCenterFingerprint;  // SIByL current_center
groupshared float3 sCurrentCenterPosition;     // SIByL current_center_pos
groupshared float  sCurrentCenterIntensity;    // SIByL current_center_intensity
groupshared int    sSelectedWarp;              // SIByL selected_cluster
// SIByL's subgroup reduction and shuffles are emulated through groupshared, NOT
// wave intrinsics. The 32 "lanes" of a logical warp here are a THREAD-INDEX
// slice, not a hardware wave: built on WaveGetLaneIndex()/WaveActiveSum() this
// kernel was correct only at wave32, and a 64-wide wave leaves half of
// sWarpProbability unwritten (WaveIsFirstLane fires once per 64) while summing
// two logical warps together. Index arithmetic cannot drift with the driver's
// wave-size choice. See the packing note in vxpgFingerprint.hlsl.
groupshared float  sSeedExchange[CLUSTER_SEED_THREADS];
groupshared int    sWarpNodeAtLaneZero[32];

[numthreads(1024, 1, 1)]
void SeedClusterCenters(uint3 tid : SV_DispatchThreadID)
{
    const uint threadId = tid.x;
    const uint laneId = threadId % 32u;      // 0..31 within the logical warp
    const uint warpId = threadId / 32u;      // 0..31 logical warp
    const uint warpBase = warpId * 32u;
    const int litVoxelCount = int(gGuidingDispatchArgs[0].w);

    // Zeroing here rather than in a node of its own: this kernel is one group and
    // always runs immediately before the assignment that fills them.
    if (gCollectStats != 0u && threadId < uint(CLUSTER_STAT_COUNT))
        gClusterStats[threadId] = 0u;

    // Same reasoning for the Lloyd sums, and it saves a clear kernel: this group
    // zeroes them once here, and every update consumes and re-zeroes its own
    // cluster's slots, so the accumulator is always clean when a round starts.
    for (uint slot = threadId; slot < uint(CLUSTER_ACC_COUNT); slot += CLUSTER_SEED_THREADS)
        gClusterAccumulators[slot] = 0u;

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
        // nearest over all picked centers.
        const float d = ClusterDistance(candidateFingerprint, sCurrentCenterFingerprint,
                                        candidatePosition, sCurrentCenterPosition,
                                        candidateIntensity, sCurrentCenterIntensity);
        nearestCenterDistance = min(nearestCenterDistance, d);

        // k-means++: selection probability ~ squared distance
        float weight = nearestCenterDistance * nearestCenterDistance;
        sSeedExchange[threadId] = weight;
        GroupMemoryBarrierWithGroupSync();
        if (laneId == 0u)
        {
            float warpWeightSum = 0.0;
            [unroll] for (uint i = 0; i < 32u; ++i)
                warpWeightSum += sSeedExchange[warpBase + i];
            sWarpProbability[warpId] = warpWeightSum;
        }

        GroupMemoryBarrierWithGroupSync();

        // Warp 0 re-purposes its lanes to hold the 32 warp sums; lane 0 zeroes
        // itself in SIByL (excludes warp 0's own sum from the top level).
        if (warpId == 0u)
        {
            weight = sWarpProbability[laneId];
            if (laneId == 0u) weight = 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // Butterfly reduction storing, per level, the probability of taking
        // the "left" child — 5 floats per thread instead of a shared 64-float
        // tree. The partner lookup goes through sSeedExchange, so the butterfly
        // spans exactly the 32 threads of the logical warp.
        // Slot 5 is never written by the reduction below (it fills 4-level, so 4..0), yet the
        // traversal's last round reads leftProbability[level + 1] with level == 4 and pushes it
        // through shared memory. Nothing consumes the result, but reading it uninitialized is
        // undefined, so it starts at the same "no information" value the reduction uses.
        float leftProbability[6];
        leftProbability[5] = 0.5;
        for (int level = 0; level < 5; ++level)
        {
            sSeedExchange[threadId] = weight;
            GroupMemoryBarrierWithGroupSync();
            const float neighborWeight = sSeedExchange[warpBase + (laneId ^ (1u << level))];
            GroupMemoryBarrierWithGroupSync();
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
            sSeedExchange[threadId] = leftProbability[level + 1];
            GroupMemoryBarrierWithGroupSync();
            leftProbability[level + 1] = sSeedExchange[warpBase + uint(clamp(nodeId, 0, 31))];
            GroupMemoryBarrierWithGroupSync();
        }

        if (threadId == 0u)
            sSelectedWarp = nodeId;
        if (laneId == 0u)
            sWarpNodeAtLaneZero[warpId] = nodeId;

        GroupMemoryBarrierWithGroupSync();

        const int selectedLane = sWarpNodeAtLaneZero[warpId];
        if (int(warpId) == sSelectedWarp && int(laneId) == selectedLane)
        {
            gClusterSeedCompactIds[seedId] = candidateId;
            ClusterCenter center;
            center.fingerprint = candidateFingerprint;
            center.position = candidatePosition;
            center.intensity = candidateIntensity;
            gClusterCenters[seedId] = center;
            // DEVIATION from SIByL: all three fields of the running center are refreshed, not
            // the fingerprint alone. Eq. (4) weights an intensity difference at weight 1, and
            // leaving that field at seed 0's value makes every later round measure a distance
            // to a center that was never picked — a constant offset instead of a separation by
            // brightness. Since there is no Lloyd iteration, the seeds ARE the cluster
            // descriptors, so the error would propagate straight into the assignment.
            sCurrentCenterFingerprint = candidateFingerprint;
            sCurrentCenterPosition    = candidatePosition;
            sCurrentCenterIntensity   = candidateIntensity;
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
    float nearestHamming = 0.0;
    float nearestIntensityTerm = 0.0;
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

            if (gCollectStats != 0u)
            {
                const uint4 diff = fingerprint ^ center.fingerprint;
                nearestHamming = countbits(diff.x) + countbits(diff.y)
                               + countbits(diff.z) + countbits(diff.w);
                nearestIntensityTerm = INTENSITY_WEIGHT * abs(intensity - center.intensity);
            }
        }
    }

    gVoxelClusterAssignments[compactId] = nearestCluster;

    if (gCollectStats != 0u && nearestCluster >= 0)
    {
        uint ignored;
        InterlockedAdd(gClusterStats[CLUSTER_STAT_POPULATION + nearestCluster], 1u, ignored);
        InterlockedAdd(gClusterStats[CLUSTER_STAT_HAMMING + nearestCluster], uint(nearestHamming), ignored);
        InterlockedAdd(gClusterStats[CLUSTER_STAT_INTENSITY + nearestCluster],
                       uint(nearestIntensityTerm * CLUSTER_STAT_INTENSITY_SCALE), ignored);
        const uint ownBits = countbits(fingerprint.x) + countbits(fingerprint.y)
                           + countbits(fingerprint.z) + countbits(fingerprint.w);
        InterlockedAdd(gClusterStats[CLUSTER_STAT_POPCOUNT + nearestCluster], ownBits, ignored);
    }
}

// ---- AccumulateClusterCenters -----------------------------------------------

// Sums one Lloyd round: every voxel adds itself to the cluster it was assigned.
// Only SET bits are touched, so the atomic traffic is the mean popcount (about
// 51 of 128 in the measured scenes) rather than the full width.
[numthreads(256, 1, 1)]
void AccumulateClusterCenters(uint3 tid : SV_DispatchThreadID)
{
    const uint litVoxelCount = gGuidingDispatchArgs[0].w;
    const uint compactId = tid.x;
    if (compactId >= litVoxelCount) return;

    const int clusterId = gVoxelClusterAssignments[compactId];
    if (clusterId < 0 || clusterId >= CLUSTER_COUNT) return;

    const uint base = uint(clusterId) * CLUSTER_ACC_STRIDE;
    const int voxelId = int(gCompactIds[compactId]);
    const float3 voxelPosition = float3(ReconstructVoxelCoord(uint(voxelId)));
    const float intensity = gPremulIrradiance[compactId];
    const uint4 fingerprint = gVoxelFingerprints[compactId];

    uint ignored;
    [unroll] for (uint word = 0; word < 4u; ++word)
    {
        uint bits = fingerprint[word];
        while (bits != 0u)
        {
            const uint bit = firstbitlow(bits);
            bits &= bits - 1u; // clear the lowest set bit
            InterlockedAdd(gClusterAccumulators[base + CLUSTER_ACC_BITS + word * 32u + bit], 1u, ignored);
        }
    }

    // Both are non-negative — voxel coordinates are grid indices and the
    // irradiance is premultiplied — so an unsigned accumulator is enough.
    [unroll] for (uint axis = 0; axis < 3u; ++axis)
        InterlockedAdd(gClusterAccumulators[base + CLUSTER_ACC_POSITION + axis],
                       uint(voxelPosition[axis] * CLUSTER_ACC_POSITION_SCALE), ignored);
    InterlockedAdd(gClusterAccumulators[base + CLUSTER_ACC_INTENSITY],
                   uint(intensity * CLUSTER_ACC_INTENSITY_SCALE), ignored);
    InterlockedAdd(gClusterAccumulators[base + CLUSTER_ACC_POPULATION], 1u, ignored);
}

// ---- UpdateClusterCenters ---------------------------------------------------

// One thread per cluster: replace the center with the centroid of its members
// and re-zero the slots for the next round.
[numthreads(CLUSTER_COUNT, 1, 1)]
void UpdateClusterCenters(uint3 tid : SV_DispatchThreadID)
{
    const uint clusterId = tid.x;
    if (clusterId >= uint(CLUSTER_COUNT)) return;

    const uint base = clusterId * CLUSTER_ACC_STRIDE;
    const uint population = gClusterAccumulators[base + CLUSTER_ACC_POPULATION];

    // An empty cluster has no centroid. Keeping its seed descriptor is what lets
    // it win members again in a later round; zeroing it would collapse the center
    // onto the origin and make the emptiness permanent.
    if (population != 0u)
    {
        ClusterCenter center;

        // Hamming's centroid is the per-bit majority: the bit that disagrees with
        // fewer members. Ties go to zero, which matches the < half test.
        uint4 fingerprint = uint4(0u, 0u, 0u, 0u);
        [unroll] for (uint word = 0; word < 4u; ++word)
        {
            uint bits = 0u;
            for (uint bit = 0; bit < 32u; ++bit)
            {
                const uint set = gClusterAccumulators[base + CLUSTER_ACC_BITS + word * 32u + bit];
                if (set * 2u > population)
                    bits |= 1u << bit;
            }
            fingerprint[word] = bits;
        }
        center.fingerprint = fingerprint;

        const float inversePopulation = 1.0 / float(population);
        center.position = float3(gClusterAccumulators[base + CLUSTER_ACC_POSITION + 0],
                                 gClusterAccumulators[base + CLUSTER_ACC_POSITION + 1],
                                 gClusterAccumulators[base + CLUSTER_ACC_POSITION + 2])
                        * (inversePopulation / CLUSTER_ACC_POSITION_SCALE);
        center.intensity = float(gClusterAccumulators[base + CLUSTER_ACC_INTENSITY])
                         * (inversePopulation / CLUSTER_ACC_INTENSITY_SCALE);

        gClusterCenters[clusterId] = center;
    }

    for (uint slot = 0; slot < uint(CLUSTER_ACC_STRIDE); ++slot)
        gClusterAccumulators[base + slot] = 0u;
}
