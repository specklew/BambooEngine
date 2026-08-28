// VXPG fingerprint pass — MRCS row sampling (Hasan et al. 2007): a handful of
// representative shading points are evaluated against every voxel, so each voxel
// ends up with the "reduced column" the cluster pass then groups on. Here that
// column is a 128-bit visibility signature. Two kernels:
//
//   SampleScreenRepresentatives  (port of svoxel/row-presample.slang)
//     Picks 128 stratified screen points (16x8 grid, one random pick per cell)
//     as stand-ins for the whole frame, and emits the downstream dispatch args.
//
//   BuildVoxelFingerprints       (port of svoxel/row-visibility.slang)
//     For each (representative pixel, lit voxel) pair traces one shadow ray;
//     the 128 visibility bits per voxel ARE its fingerprint. Voxels with similar
//     fingerprints light the same screen regions -> clustered together later.
//
// Ported from SIByL; identifiers renamed to descriptive Bamboo names (original
// SIByL names kept in comments for traceability).

#include "PassRegisters.h"
#include "Octahedral.hlsl"
#include "Random.hlsl"

#define FINGERPRINT_REPRESENTATIVE_COUNT 128  // 16 x 8 stratified screen samples
#define FINGERPRINT_MASK_WORDS 4              // 128 bits / 32
#define FINGERPRINT_ROWS_PER_GROUP 8          // numthreads.y: voxels handled per group
// Retries inside the stratification cell, then anywhere on screen. A cell that is
// entirely background cannot be rescued locally, which is what the global pass
// below is for. (An earlier note here blamed the staircase view's sky rows for
// pinning 64 of 128 representatives; that reading came from the wave-width bug in
// the packing below, which halved every reported count. Measured 2026-08-23 after
// the fix: staircase resolves 127 of 128 with cell retries alone and 128 with the
// global pass.)
#define FINGERPRINT_PRESAMPLE_ATTEMPTS 16u
#define FINGERPRINT_PRESAMPLE_GLOBAL_ATTEMPTS 16u

// Primary-hit G-buffer from light injection: .xyz world position, .w octahedral
// normal (bit-cast). Invalid (sky) pixels carry the 1e30 sentinel.
RWTexture2D<float4> gShadingPoints : BAMBOO_PASS_UAV(FINGERPRINT_PRESAMPLE_REG_SHADING_POINTS);

// ---- SampleScreenRepresentatives outputs ----------------------------------
// SIByL u_RepresentPixel: the 128 chosen surface points (pos + octa normal).
RWStructuredBuffer<float4> gScreenRepresentativePoints : BAMBOO_PASS_UAV(FINGERPRINT_PRESAMPLE_REG_REPRESENTATIVES);
// SIByL u_IndirectArgs: GPU-computed dispatch dimensions for the guiding passes
// downstream; .w of each entry carries the raw lit-voxel count.
RWStructuredBuffer<uint4> gGuidingDispatchArgs : BAMBOO_PASS_UAV(FINGERPRINT_PRESAMPLE_REG_DISPATCH_ARGS);
// SIByL u_vplCounter: [0] = compacted lit-voxel count (VoxelGuidingBuildPass).
RWStructuredBuffer<uint> gGuidingCounters : BAMBOO_PASS_UAV(FINGERPRINT_PRESAMPLE_REG_COUNTERS);

cbuffer PresampleCB : BAMBOO_PASS_CBV(FINGERPRINT_PRESAMPLE_REG_CB)
{
    uint2 gResolution;
    uint  gRandSeed;
    // 0 = one blind pick per cell, the shape this was ported in; 1 = retry until
    // the pick lands on a surface (vxpg.fingerprint.retryPresample).
    uint  gRetryPresample;
}

[numthreads(16, 8, 1)]
void SampleScreenRepresentatives(uint3 tid : SV_DispatchThreadID)
{
    const uint2 cellId = tid.xy; // 16 x 8 grid cell = one representative
    const uint flattenId = cellId.y * 16u + cellId.x;

    // Stratified jitter, retried until the pick lands on a surface. A single
    // blind pick spends the representative on the sky whenever the cell is mostly
    // background, and a sky representative is not a receiver: it contributes no
    // bit to any voxel's fingerprint, so the 128-bit signature silently narrows.
    // Measured 2026-08-23 with the loop disabled
    // (vxpg.fingerprint.retryPresample=0): 18 of 128 valid on ABeautifulGame and
    // 127 of 128 on the staircase. With it on: 120 and 128. ABeautifulGame is
    // where the stratified blind pick was throwing the clustering signal away.
    const uint seed = pcg_hash((flattenId * 9781u + gRandSeed * 26699u) | 1u);
    const float2 cellSize = float2(gResolution) / float2(16.0, 8.0);

    const uint cellAttempts = (gRetryPresample != 0u) ? FINGERPRINT_PRESAMPLE_ATTEMPTS : 1u;

    float4 representative = float4(0, 0, 0, 0); // all-background cell: no receiver
    [loop] for (uint attempt = 0; attempt < cellAttempts; ++attempt)
    {
        const float2 jitter = Random2D(pcg_hash(seed + attempt * 7919u));
        const float2 samplePixel = cellSize * (float2(cellId) + jitter);
        const int2 pixelInt = clamp(int2(samplePixel), int2(0, 0), int2(gResolution) - int2(1, 1));

        const float4 candidate = gShadingPoints[pixelInt];
        if (!any(candidate >= 1e30))
        {
            representative = candidate;
            break;
        }
    }

    // Cell exhausted: take the stratification loss rather than the dead bit and
    // look anywhere on screen. A uniformly placed receiver still carries real
    // visibility information; an unfilled one carries none.
    if (gRetryPresample != 0u && all(representative == 0.0))
    {
        [loop] for (uint globalAttempt = 0; globalAttempt < FINGERPRINT_PRESAMPLE_GLOBAL_ATTEMPTS; ++globalAttempt)
        {
            const float2 anywhere = Random2D(pcg_hash(seed + 104729u + globalAttempt * 15485863u));
            const int2 pixelInt = clamp(int2(anywhere * float2(gResolution)),
                                        int2(0, 0), int2(gResolution) - int2(1, 1));

            const float4 candidate = gShadingPoints[pixelInt];
            if (!any(candidate >= 1e30))
            {
                representative = candidate;
                break;
            }
        }
    }

    gScreenRepresentativePoints[flattenId] = representative;

    // Thread (0,0) alone emits the dispatch args (SIByL row-presample tail).
    if (all(cellId == uint2(0, 0)))
    {
        const uint litVoxelCount = gGuidingCounters[0];
        gGuidingDispatchArgs[0] = uint4((litVoxelCount + 255u) / 256u, 1u, 1u, litVoxelCount);
        gGuidingDispatchArgs[1] = uint4((litVoxelCount * 2u - 1u + 255u) / 256u, 1u, 1u, litVoxelCount);
        // Row-visibility grid: X covers 128 representatives (4 groups of 32),
        // Y covers voxels (8 per group).
        gGuidingDispatchArgs[2] = uint4(4u, (litVoxelCount + 7u) / 8u, 1u, litVoxelCount);
    }
}

// ---- BuildVoxelFingerprints -----------------------------------------------

RaytracingAccelerationStructure gSceneBVH : BAMBOO_PASS_SRV(FINGERPRINT_VISIBILITY_REG_TLAS);

// SIByL u_RepresentPixel / u_RepresentVPL / u_IndirectArgs. UAV-typed reads:
// Bamboo keeps these buffers in UNORDERED_ACCESS state so no SRV transition is
// needed between the presample and visibility kernels.
RWStructuredBuffer<float4> gReadRepresentativePoints : BAMBOO_PASS_UAV(FINGERPRINT_VISIBILITY_REG_REPRESENTATIVES);
RWStructuredBuffer<float4> gCompactVoxelLightPoints  : BAMBOO_PASS_UAV(FINGERPRINT_VISIBILITY_REG_LIGHT_POINTS);
RWStructuredBuffer<uint4>  gReadDispatchArgs         : BAMBOO_PASS_UAV(FINGERPRINT_VISIBILITY_REG_DISPATCH_ARGS);

// SIByL u_RowVisibility: 4 uints per compact voxel = the 128-bit fingerprint.
RWStructuredBuffer<uint> gVoxelFingerprints : BAMBOO_PASS_UAV(FINGERPRINT_VISIBILITY_REG_FINGERPRINTS);

static const float FINGERPRINT_RAY_EPSILON = 0.01;

// Diagnostic decomposition of "not visible" (vxpg.fingerprint.probe). A voxel's
// popcount is the product of three independent gates, and the measured value is
// tiny; these modes report each gate on its own so the responsible one is a
// number rather than a guess. Read the result through vxpg.cluster.dumpStats'
// mean-popcount line.
#define FINGERPRINT_PROBE_NONE            0 // the real test
#define FINGERPRINT_PROBE_NO_FACING       1 // occlusion only
#define FINGERPRINT_PROBE_NO_OCCLUSION    2 // facing only
#define FINGERPRINT_PROBE_RECEIVER_VALID  3 // how many representatives hit a surface at all
// Mode 4 tests the PACKING, not the scene: every live lane votes visible, so a
// correct pass reports a mean popcount of exactly 128.0 on any scene with at
// least one lit voxel. Anything lower means mask words are not reaching their
// voxel — which is how the 2026-08-23 wave-width bug presented (a flat 64.0,
// half the voxels never written). Run it as:
//   --cvar vxpg.fingerprint.probe=4 --cvar vxpg.cluster.dumpStats=1
// and read the "mean fingerprint popcount" line.
#define FINGERPRINT_PROBE_PACKING         4 // structural self-test: must report 128.0

cbuffer VisibilityCB : BAMBOO_PASS_CBV(FINGERPRINT_VISIBILITY_REG_CB)
{
    uint gFingerprintProbe;
    uint _visibilityPad0;
    uint _visibilityPad1;
    uint _visibilityPad2;
}

// One 32-bit mask word per (voxel row, representative group). Packed through
// groupshared rather than WaveActiveBallot: a ballot's bit i is LANE i, and a
// wave wider than the group's x extent straddles two voxel rows, so lane 32+
// would vote into a neighbouring voxel's word and WaveIsFirstLane would leave
// that voxel unwritten. numthreads.x is the packing width by construction here,
// which no driver wave-size choice can change.
groupshared uint sFingerprintWord[FINGERPRINT_ROWS_PER_GROUP];

[numthreads(32, 8, 1)]
void BuildVoxelFingerprints(uint3 tid : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint row = groupThreadId.y;
    const uint bit = groupThreadId.x;        // 0..31 within this representative word
    if (bit == 0u)
        sFingerprintWord[row] = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint compactID = tid.y;            // lit voxel (SIByL DTid.y)
    const uint litVoxelCount = gReadDispatchArgs[0].w;
    // Over-dispatched rows fold into the vote as "not visible" instead of
    // returning: the group-wide barriers below must stay uniform.
    const bool rowIsLive = compactID < litVoxelCount;

    const float4 lightPoint = rowIsLive ? gCompactVoxelLightPoints[compactID] : float4(0, 0, 0, 0);
    const float3 lightPointPosition = lightPoint.xyz;
    const float3 lightPointNormal = Unorm32OctahedronToUnitVector(asuint(lightPoint.w));

    const uint representativeIndex = tid.x;  // 0..127 (SIByL DTid.x)
    const float4 receiverPoint = gReadRepresentativePoints[representativeIndex];
    const float3 receiverPosition = receiverPoint.xyz;
    const float3 receiverNormal = Unorm32OctahedronToUnitVector(asuint(receiverPoint.w));

    bool visible = true;

    // A representative that landed on the sky was zeroed by the presample, which
    // puts a fake receiver at the world origin — it must not be treated as a
    // surface that can see anything.
    const bool receiverIsValid = any(receiverPoint != 0.0);

    // Facing test both ways: light can't leave a surface backward, nor arrive
    // from behind the receiver.
    float3 toReceiver = receiverPosition - lightPointPosition;
    const float distance = length(toReceiver);
    toReceiver /= max(distance, 1e-8);
    const bool hasDistance    = distance > 1e-6;
    const bool facesEachOther = hasDistance &&
                                dot(lightPointNormal, toReceiver) >= 0.0 &&
                                dot(receiverNormal, -toReceiver) >= 0.0;

    const bool probeReceiverValid = (gFingerprintProbe == FINGERPRINT_PROBE_RECEIVER_VALID);
    const bool probePacking       = (gFingerprintProbe == FINGERPRINT_PROBE_PACKING);

    visible = rowIsLive &&
              (probePacking ||
               (receiverIsValid &&
                (probeReceiverValid ||
                 (hasDistance && (facesEachOther || gFingerprintProbe == FINGERPRINT_PROBE_NO_FACING)))));

    if (visible && !probeReceiverValid && !probePacking &&
        gFingerprintProbe != FINGERPRINT_PROBE_NO_OCCLUSION)
    {
        RayDesc ray;
        ray.Origin = lightPointPosition + lightPointNormal * FINGERPRINT_RAY_EPSILON;
        ray.Direction = toReceiver;
        ray.TMin = 0.0;
        ray.TMax = max(FINGERPRINT_RAY_EPSILON, distance - 2.0 * FINGERPRINT_RAY_EPSILON);

        // FORCE_OPAQUE states what this test already did: alpha-cutout geometry counts
        // as a full occluder. Deliberate coarsening — the fingerprint only decides which
        // voxels cluster together, so a miss through foliage costs guiding efficiency,
        // never correctness, and alpha-testing it would drag the instance/geometry/texture
        // bindings into this pass for a 128-representative inner loop. The shadow rays
        // that DO carry radiance (raytracing.shadow.hlsl) alpha-test properly.
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(gSceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
        q.Proceed();
        if (q.CommittedStatus() != COMMITTED_NOTHING)
            visible = false;
    }

    if (visible)
        InterlockedOr(sFingerprintWord[row], 1u << bit);
    GroupMemoryBarrierWithGroupSync();

    if (bit == 0u && rowIsLive)
    {
        const uint maskWordIndex = representativeIndex / 32u;
        gVoxelFingerprints[compactID * FINGERPRINT_MASK_WORDS + maskWordIndex] = sFingerprintWord[row];
    }
}
