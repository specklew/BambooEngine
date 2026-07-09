// VXPG cluster-visibility pass (MRCS "C-lean" soft visibility): fills the
// per-superpixel x per-cluster visibility matrix that makes guiding view-
// adaptive. Three kernels run each frame after the superpixel + cluster passes:
//
//   ClearClusterVisibility  (port of visibility/cvis-clear.slang)
//     Zeroes the mask texture, cluster light-point counts, and the soft
//     avg-visibility buffer.
//
//   GatherClusterLightPoints (port of visibility/cvis-assignment-ex.slang)
//     One thread per pixel: files the pixel's bounce VPL into its voxel's
//     cluster drawer, and seeds the mask bit for the (superpixel, cluster)
//     connection the pixel's own path already proved.
//
//   CheckClusterVisibility  (port of visibility/cvis-visibility-check-comp.slang)
//     Per (superpixel, cluster): 32 sample lanes each trace one shadow ray
//     from a random superpixel pixel to a random cluster VPL; the BRDF-weighted
//     visible fraction is the soft avg-visibility, and any hit sets the mask.
//
// Ported from SIByL; identifiers renamed to descriptive Bamboo names (original
// SIByL names in comments). Substitution: Cook-Torrance replaces SIByL's
// Lambertian/RoughPlastic BSDF weight (Bamboo's material family).

#include "RaytracingUtils.hlsl" // scene bindings + VBuffer hit reconstruction
#include "BRDF.hlsl"
#include "Octahedral.hlsl"
#include "VBuffer.hlsl"

#define CVIS_CLUSTER_COUNT 32
#define CVIS_GATHER_CAP    1024

// ---- cvis-specific resources (registers chosen to avoid RaytracingUtils') ----
// Table (global heap): per-pixel + per-superpixel textures.
RWTexture2D<float4> gVplPosition             : register(u1); // SIByL u_vpl_position (slot 526)
RWTexture2D<uint4>  gVBuffer                 : register(u2); // shared primary VBuffer (slot 527)
RWTexture2D<int>    gSuperpixelIndex         : register(u3); // SIByL u_spixelIdx (slot 523)
RWTexture2D<int2>   gSpixelGathered          : register(u4); // SIByL u_spixel_gathered (slot 528)
RWTexture2D<uint>   gSpixelCounter           : register(u5); // SIByL u_spixel_counter (slot 529)
RWTexture2D<uint>   gClusterVisibilityMask   : register(u6); // SIByL u_spixel_visibility (slot 530)

// Root UAVs / SRVs: structured buffers.
RWStructuredBuffer<int>    gVoxInverseIndex          : register(u7);  // voxelID -> compactID
RWStructuredBuffer<int>    gVoxelClusterAssignments  : register(u8);  // SIByL u_associate_buffer
RWStructuredBuffer<float4> gClusterGatheredLightPoints : register(u9); // SIByL u_cluster_gathered
RWStructuredBuffer<uint>   gClusterLightPointCounts  : register(u10); // SIByL u_cluster_counter
RWStructuredBuffer<float>  gSpixelClusterAvgVisibility : register(u11); // SIByL u_spixel_avg_visibility

cbuffer CvisGridCB : register(b1)
{
    float3 gGridMin;
    float  gVoxelSize;
    float3 gGridMax;
    uint   gGridDim;
}

cbuffer CvisCB : register(b2)
{
    int2 gResolution;
    int2 gMapSize;      // (mapX, mapY) = superpixel grid dimensions
    uint gSeed;         // frame index (frame-varying, faithful)
    uint gUseBsdf;      // vxpg.cvis.useBsdf
    uint gUseDistance;  // vxpg.cvis.useDistance
    uint gInstanceCount; // guards VBuffer instanceId against g_instanceInfo size
}

int VoxelFlatId(float3 worldPos)
{
    int3 v = int3(floor((worldPos - gGridMin) / gVoxelSize));
    if (any(v < 0) || any(v >= int(gGridDim))) return -1;
    return v.x + v.y * int(gGridDim) + v.z * int(gGridDim) * int(gGridDim);
}

// ---- ClearClusterVisibility (cvis-clear) ----------------------------------

[numthreads(16, 16, 1)]
void ClearClusterVisibility(uint3 tid : SV_DispatchThreadID)
{
    const int2 sp = int2(tid.xy);
    if (any(sp >= gMapSize)) return;

    gClusterVisibilityMask[sp] = 0u;

    const uint spixelFlat = uint(sp.y) * uint(gMapSize.x) + uint(sp.x);
    [unroll] for (uint c = 0; c < CVIS_CLUSTER_COUNT; ++c)
        gSpixelClusterAvgVisibility[spixelFlat * CVIS_CLUSTER_COUNT + c] = 0.0;

    // The 32 cluster counters ride the first row of the dispatch.
    if (sp.x < CVIS_CLUSTER_COUNT && sp.y == 0)
        gClusterLightPointCounts[sp.x] = 0u;
}

// ---- GatherClusterLightPoints (cvis-assignment-ex) ------------------------

[numthreads(16, 16, 1)]
void GatherClusterLightPoints(uint3 tid : SV_DispatchThreadID)
{
    const int2 pixelID = int2(tid.xy);
    if (any(pixelID >= gResolution)) return;

    const float4 vpl = gVplPosition[pixelID];
    if (all(vpl == float4(0, 0, 0, 0))) return; // bounce missed (sky)

    const int flatId = VoxelFlatId(vpl.xyz);
    if (flatId < 0) return;
    const int compactID = gVoxInverseIndex[flatId];
    if (compactID < 0) return;
    const int clusterID = gVoxelClusterAssignments[compactID];
    if (clusterID < 0 || clusterID >= CVIS_CLUSTER_COUNT) return;

    // File this VPL into the cluster's drawer (deli-ticket append).
    uint slot;
    InterlockedAdd(gClusterLightPointCounts[clusterID], 1u, slot);
    if (slot < CVIS_GATHER_CAP)
        gClusterGatheredLightPoints[clusterID * CVIS_GATHER_CAP + slot] = vpl;

    // Free seed: this pixel's own path reached the VPL, so its superpixel
    // provably sees this cluster.
    const int spixelID = gSuperpixelIndex[pixelID];
    if (spixelID < 0) return;
    const int2 sp2D = int2(spixelID % gMapSize.x, spixelID / gMapSize.x);
    InterlockedOr(gClusterVisibilityMask[sp2D], 1u << uint(clusterID));
}

// ---- CheckClusterVisibility (cvis-visibility-check-comp) -------------------

float NextRandom(inout uint state)
{
    state = pcg_hash(state);
    return float(state) * (1.0 / 4294967296.0);
}

// maxComponent(Cook-Torrance BRDF) as the SIByL-equivalent BSDF weight.
// DEVIATION: uses per-instance material FACTORS (base color / roughness /
// metallic) + the geometric normal, NOT the albedo/roughness/normal textures.
// A plain compute Dispatch can't sample the scene textures while they sit in
// the raster path's PIXEL_SHADER_RESOURCE layout; the RT passes (DispatchRays)
// tolerate it but this compute pass does not. The weight is a soft visibility
// refinement, so instance constants are an acceptable stand-in.
float BsdfWeight(int2 pixelID, float3 lightDir)
{
    VBufferData vb = UnpackVBufferData(gVBuffer[pixelID]);
    if (IsVBufferInvalid(vb) || vb.instanceId >= gInstanceCount) return 0.0;

    InstanceInfo instance = g_instanceInfo[vb.instanceId];
    GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
    HitData hit = GetHitData(vb.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                             vb.barycentrics, instance.objectToWorld);

    float3 albedo = instance.baseColorFactor.rgb;
    float roughness = max(instance.roughnessFactor, MIN_ROUGHNESS);
    float metallic = instance.metallicFactor;

    float3 camPos = mul(viewI, float4(0, 0, 0, 1)).xyz;
    float3 V = normalize(camPos - hit.position);
    float3 N = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
    if (dot(N, V) < 0.0) N = -N;

    float NdotL = dot(N, lightDir);
    if (NdotL <= 0.0) return 0.0;
    float NdotV = max(dot(N, V), 1e-3);

    float3 H = normalize(V + lightDir);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float  D = DistributionGGX(NdotH, roughness);
    float  G = SmithG_GGX(NdotV, NdotL, roughness);
    float3 F = FresnelSchlick(VdotH, F0);
    float3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 brdf = (kD * albedo / PI + specular) * NdotL;

    return max(brdf.r, max(brdf.g, brdf.b));
}

groupshared float sSharedVisibility[CVIS_CLUSTER_COUNT];
groupshared uint  sGroupVisibility;

[numthreads(32, 8, 1)]
[WaveSize(32)]
void CheckClusterVisibility(uint3 dtid : SV_DispatchThreadID, uint gidx : SV_GroupIndex)
{
    const int2 spixelID = int2(dtid.xy) / 32;
    const int2 task = int2(dtid.xy) % 32;
    const int  clusterToCheck = task.y;

    if (gidx < CVIS_CLUSTER_COUNT)
    {
        sSharedVisibility[gidx] = 0.0;
        if (gidx == 0) sGroupVisibility = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    float weight = 0.0;
    bool visible = false;

    // Per-lane unique seed so the 32 lanes pick different pixel/VPL pairs.
    uint rng = pcg_hash((uint(dtid.x) * 1973u + uint(dtid.y) * 9277u + gSeed * 26699u) | 1u);

    const uint spixelCount = clamp(gSpixelCounter[spixelID], 0u, uint(CVIS_GATHER_CAP));
    const uint clusterVplCount = clamp(gClusterLightPointCounts[clusterToCheck], 0u, uint(CVIS_GATHER_CAP));

    if (spixelCount > 0u && clusterVplCount > 0u)
    {
        const uint pixelSlot = min(uint(NextRandom(rng) * spixelCount), spixelCount - 1u);
        const int2 subtask = int2(int(pixelSlot) % 32, int(pixelSlot) / 32);
        // Gather lists can hold stale/unwritten entries; clamp to a valid screen
        // pixel so a bogus coordinate can't read the VBuffer (and instance info)
        // out of bounds.
        // Gather lists can hold stale/unwritten entries; clamp to a valid screen
        // pixel so a bogus coordinate can't read the VBuffer out of bounds.
        const int2 pixelID = clamp(gSpixelGathered[spixelID * 32 + subtask],
                                   int2(0, 0), gResolution - int2(1, 1));

        VBufferData vb = UnpackVBufferData(gVBuffer[pixelID]);
        if (!IsVBufferInvalid(vb) && vb.instanceId < gInstanceCount)
        {
            InstanceInfo instance = g_instanceInfo[vb.instanceId];
            GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
            HitData hit = GetHitData(vb.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                                     vb.barycentrics, instance.objectToWorld);
            float3 hitNormal = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
            // Face-forward against the camera ray, as SIByL's GetGeometryHit(vhit,
            // ray) does before its facing gate (spt_implement.hlsli:99-102) — the
            // port dropped this flip. With the raw winding normal, back-wound
            // surfaces (Sponza vault undersides, arcade ceilings) failed the gate
            // for EVERY VPL, zeroing those superpixels' entire visibility rows:
            // dead guide across ~35-40% of the frame plus frame-to-frame flicker
            // where the sampled pixel's winding changed per frame.
            float3 cameraPos = mul(viewI, float4(0, 0, 0, 1)).xyz;
            if (dot(hitNormal, normalize(cameraPos - hit.position)) < 0.0)
                hitNormal = -hitNormal;

            const uint vplSlot = min(uint(NextRandom(rng) * clusterVplCount), clusterVplCount - 1u);
            const float4 vplPosNorm = gClusterGatheredLightPoints[clusterToCheck * CVIS_GATHER_CAP + vplSlot];
            const float3 vplNormal = Unorm32OctahedronToUnitVector(asuint(vplPosNorm.w));

            float3 dir = vplPosNorm.xyz - hit.position;
            const float dist = length(dir);
            dir /= max(dist, 1e-8);

            if (dot(-dir, vplNormal) > 1e-4 && dot(dir, hitNormal) > 1e-4)
            {
                RayDesc ray;
                ray.Origin = hit.position + hitNormal * 0.01;
                ray.Direction = dir;
                ray.TMin = 0.0;
                ray.TMax = max(0.01, dist - 0.02);

                RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                q.TraceRayInline(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, ray);
                while (q.Proceed())
                    q.CommitNonOpaqueTriangleHit();
                visible = (q.CommittedStatus() == COMMITTED_NOTHING);

                if (visible)
                {
                    weight = (gUseBsdf != 0u) ? BsdfWeight(pixelID, dir) : 1.0;
                    if (gUseDistance != 0u)
                        weight /= (dist * dist);
                }
            }
        }
    }

    // 32 sample lanes (one wave, fixed clusterToCheck) sum their weights.
    float waveSum = WaveActiveSum(visible ? weight : 0.0);
    if (WaveIsFirstLane())
        sSharedVisibility[clusterToCheck] = waveSum;

    GroupMemoryBarrierWithGroupSync();

    // This group covers clusters [clusterOffset, clusterOffset+8); each of those
    // 8 lanes writes its cluster's avg entry and ORs the mask bit.
    if (gidx < CVIS_CLUSTER_COUNT)
    {
        const uint spixelFlat = uint(spixelID.y) * uint(gMapSize.x) + uint(spixelID.x);
        const int clusterOffset = (clusterToCheck / 8) * 8;
        if (int(gidx) >= clusterOffset && int(gidx) < clusterOffset + 8)
        {
            float v = sSharedVisibility[gidx];
            gSpixelClusterAvgVisibility[spixelFlat * CVIS_CLUSTER_COUNT + gidx] = v;
            if (v != 0.0)
                InterlockedOr(sGroupVisibility, 1u << gidx);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (gidx == 0)
        InterlockedOr(gClusterVisibilityMask[spixelID], sGroupVisibility);
}
