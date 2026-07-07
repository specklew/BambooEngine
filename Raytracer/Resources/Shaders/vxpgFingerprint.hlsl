// VXPG fingerprint pass — the MRCS "column reduction" that gives every lit
// voxel a 128-bit visibility signature. Two kernels:
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

#include "Octahedral.hlsl"
#include "Random.hlsl"

#define FINGERPRINT_REPRESENTATIVE_COUNT 128  // 16 x 8 stratified screen samples
#define FINGERPRINT_MASK_WORDS 4              // 128 bits / 32

// Primary-hit G-buffer from light injection: .xyz world position, .w octahedral
// normal (bit-cast). Invalid (sky) pixels carry the 1e30 sentinel.
RWTexture2D<float4> gShadingPoints : register(u0);

// ---- SampleScreenRepresentatives outputs ----------------------------------
// SIByL u_RepresentPixel: the 128 chosen surface points (pos + octa normal).
RWStructuredBuffer<float4> gScreenRepresentativePoints : register(u1);
// SIByL u_IndirectArgs: GPU-computed dispatch dimensions for the guiding passes
// downstream; .w of each entry carries the raw lit-voxel count.
RWStructuredBuffer<uint4> gGuidingDispatchArgs : register(u2);
// SIByL u_vplCounter: [0] = compacted lit-voxel count (VoxelGuidingBuildPass).
RWStructuredBuffer<uint> gGuidingCounters : register(u3);

cbuffer PresampleCB : register(b0)
{
    uint2 gResolution;
    uint  gRandSeed;
    uint  _presamplePad0;
}

[numthreads(16, 8, 1)]
void SampleScreenRepresentatives(uint3 tid : SV_DispatchThreadID)
{
    const uint2 cellId = tid.xy; // 16 x 8 grid cell = one representative
    const uint flattenId = cellId.y * 16u + cellId.x;

    // Stratified jitter: one random point inside this cell.
    uint seed = pcg_hash((flattenId * 9781u + gRandSeed * 26699u) | 1u);
    float2 jitter = Random2D(seed);
    float2 cellSize = float2(gResolution) / float2(16.0, 8.0);
    float2 samplePixel = cellSize * (float2(cellId) + jitter);
    int2 pixelInt = clamp(int2(samplePixel), int2(0, 0), int2(gResolution) - int2(1, 1));

    float4 representative = gShadingPoints[pixelInt];
    if (any(representative >= 1e30)) representative = float4(0, 0, 0, 0); // sky/no-hit
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

RaytracingAccelerationStructure gSceneBVH : register(t0);

// SIByL u_RepresentPixel / u_RepresentVPL / u_IndirectArgs. UAV-typed reads:
// Bamboo keeps these buffers in UNORDERED_ACCESS state so no SRV transition is
// needed between the presample and visibility kernels.
RWStructuredBuffer<float4> gReadRepresentativePoints : register(u1);
RWStructuredBuffer<float4> gCompactVoxelLightPoints  : register(u2);
RWStructuredBuffer<uint4>  gReadDispatchArgs         : register(u3);

// SIByL u_RowVisibility: 4 uints per compact voxel = the 128-bit fingerprint.
RWStructuredBuffer<uint> gVoxelFingerprints : register(u4);

static const float FINGERPRINT_RAY_EPSILON = 0.01;

[numthreads(32, 8, 1)]
[WaveSize(32)]
void BuildVoxelFingerprints(uint3 tid : SV_DispatchThreadID)
{
    const uint compactID = tid.y;            // lit voxel (SIByL DTid.y)
    const uint litVoxelCount = gReadDispatchArgs[0].w;
    if (compactID >= litVoxelCount) return;  // over-dispatch early-out (option b)

    const float4 lightPoint = gCompactVoxelLightPoints[compactID];
    const float3 lightPointPosition = lightPoint.xyz;
    const float3 lightPointNormal = Unorm32OctahedronToUnitVector(asuint(lightPoint.w));

    const uint representativeIndex = tid.x;  // 0..127 (SIByL DTid.x)
    const float4 receiverPoint = gReadRepresentativePoints[representativeIndex];
    const float3 receiverPosition = receiverPoint.xyz;
    const float3 receiverNormal = Unorm32OctahedronToUnitVector(asuint(receiverPoint.w));

    bool visible = true;

    // Facing test both ways: light can't leave a surface backward, nor arrive
    // from behind the receiver. Also rejects the zeroed sky representatives.
    float3 toReceiver = receiverPosition - lightPointPosition;
    const float distance = length(toReceiver);
    toReceiver /= max(distance, 1e-8);
    if (distance <= 1e-6 ||
        dot(lightPointNormal, toReceiver) < 0.0 ||
        dot(receiverNormal, -toReceiver) < 0.0)
    {
        visible = false;
    }

    if (visible)
    {
        RayDesc ray;
        ray.Origin = lightPointPosition + lightPointNormal * FINGERPRINT_RAY_EPSILON;
        ray.Direction = toReceiver;
        ray.TMin = 0.0;
        ray.TMax = max(FINGERPRINT_RAY_EPSILON, distance - 2.0 * FINGERPRINT_RAY_EPSILON);

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(gSceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, ray);
        // Treat any geometry (opaque or alpha-cutout) as an occluder — a coarse
        // clustering signal, matching SIByL's opaque-only visibility test.
        while (q.Proceed())
            q.CommitNonOpaqueTriangleHit();
        if (q.CommittedStatus() != COMMITTED_NOTHING)
            visible = false;
    }

    // 32 lanes (one representative-word) vote; lane 0 writes the packed word.
    const uint4 visibilityMask = WaveActiveBallot(visible);
    if (WaveIsFirstLane())
    {
        const uint maskWordIndex = representativeIndex / 32u;
        gVoxelFingerprints[compactID * FINGERPRINT_MASK_WORDS + maskWordIndex] = visibilityMask.x;
    }
}
