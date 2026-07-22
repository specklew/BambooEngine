#ifndef GUIDED_PATH_TRACING_HLSL
#define GUIDED_PATH_TRACING_HLSL

// VXPG guided path tracing: two-sample MIS at the first bounce between BSDF
// sampling and the tree-backed voxel guide (vxguiding-gi.slang, strategy 5
// "EXT"). Forward: superpixel -> per-superpixel importance heap (5 binary
// decisions -> cluster + pdf_top) -> cluster-root light-tree walk by intensity
// ratios (-> voxel + pdf_tree) -> exact solid-angle sampling of the voxel's
// visible AABB faces as spherical quads (pdf_dir; SIByL SampleSphericalVoxel —
// the cone-toward-bounding-sphere deviation was reversed 2026-07-09, ADR 0003).
// Reverse pdf for the BSDF sample telescopes to two divisions (heap leaf/total
// x tree leaf/root intensity) via the inverse index / cluster assignments /
// compact->leaf maps. pdf products ride in GuidePdf (see GUIDE_PDF_FP64 below;
// SIByL uses double, DoublePrecisionFloatShaderOps still hard-checked at init).
//
// Guiding is applied at the FIRST vertex only; deeper bounces are vanilla
// recursion (deviation: SIByL's degraded second-bounce bolt-on omitted,
// ADR 0003). When the guide is dead for a pixel (superpixel heap root 0 or
// no lit voxels) the pixel is BSDF-only that frame at full MIS weight —
// SIByL-faithful; the old stage-A uniform-sphere fallback is gone.
//
// Reuses scene bindings and BRDF helpers from raytracing.hlsl.
#include "raytracing.hlsl"
#include "Octahedral.hlsl"
#include "VBuffer.hlsl"
#include "LightTreeNode.hlsl"
#include "SphericalQuad.hlsl"

// 1 = guiding debug views 1-14 compiled in (default; interactive debug
// variant), 0 = clean benchmark variant (guidedPathTracing.rg.clean.shader):
// debugView folds to constant 0, so every view branch and the view-5..14
// blocks drop out of the raygen entirely.
#ifndef GUIDING_DEBUG_VIEWS
#define GUIDING_DEBUG_VIEWS 1
#endif

// Guide-pdf precision switch. SIByL carries the pdf chain in double; consumer
// GPUs run FP64 at 1/16 (RDNA) to 1/64 (GeForce) of FP32 rate and doubles
// double the register footprint of the raygen. Float survives the telescoped
// chain (worst-case leaf/root ~1e-9 power-squared ~= 1e-18 vs float's ~1e-38
// floor, ADR 0003). 1 = SIByL-faithful double, 0 = float (measured deviation).
#define GUIDE_PDF_FP64 0
#if GUIDE_PDF_FP64
typedef double GuidePdf;
#else
typedef float GuidePdf;
#endif

// ---- VXPG guiding resources ----

RWTexture3D<uint> gVoxIrradiance : register(u1);
RWTexture3D<uint> gVoxVplCount   : register(u2);

// [0] = compacted voxel count ([1] retired with the flat CDF)
RWStructuredBuffer<uint>  gVoxCounters     : register(u3);
RWStructuredBuffer<uint>  gVoxCompactIds   : register(u4);
// Per-pixel SLIC superpixel assignment (flat map index, -1 invalid), written by
// SuperpixelBuildPass. SIByL u_spixelIdx. NOT pixel/32 — assignment follows
// geometry. Global-heap slot 523.
RWTexture2D<int> gSpixelIndexImage : register(u5);
// voxelID (flat) -> compactID, sentinel -1 (built by VoxelGuidingBuildPass).
RWStructuredBuffer<int>   gVoxInverseIndex : register(u6);

// Written by light injection (debug views 6/7 read them here): per-voxel
// representative VPL and per-pixel VPL hit position, both pos + octa normal.
RWTexture3D<float4> gVoxelRepresentative : register(u7);
RWTexture2D<float4> gVplPosition         : register(u8);

// Shared primary-visibility buffer (ADR 0004): the first path vertex comes
// from here; all spp samples of a frame share it and diverge at the bounce.
RWTexture2D<uint4> gVBuffer : register(u9);

// Voxel fingerprints (4 uints = 128-bit visibility mask per compact voxel),
// built by VxpgFingerprintPass. Read only by debug view 8.
RWStructuredBuffer<uint> gVoxelFingerprints : register(u10);

// Cluster pass outputs (debug view 9). SIByL u_Clusters / u_Seeds.
RWStructuredBuffer<int> gVoxelClusterAssignments : register(u11);
RWStructuredBuffer<int> gClusterSeedCompactIds   : register(u12);

// Cluster-visibility mask (debug view 10). SIByL u_spixel_visibility: bit k =
// this superpixel tile can see light cluster k. Global-heap slot 530.
RWTexture2D<uint> gClusterVisibilityMask : register(u13);

// Bottom light tree (guided sampling + debug view 11). SIByL u_Nodes /
// compact2leaf / cluster_roots.
RWStructuredBuffer<LightTreeNode> gLightTreeNodes  : register(u14);
RWStructuredBuffer<int>           gCompactToLeaf   : register(u15);
RWStructuredBuffer<int>           gClusterRootNodes : register(u16);

// Top-level tree (guided sampling + debug view 12). SIByL tltree:
// per-superpixel 64-slot importance heap.
RWStructuredBuffer<float>         gSpixelClusterImportanceHeap : register(u17);

// Live per-voxel geometry bounds, quantized to the voxel cube (uint 0 =
// cube min, 0xffffffff = cube max), reloaded from the bake each frame by
// VoxelGuidingBuildPass. With voxel.bake.useCompact off (default) the bake
// stores the full cube, so unpacking is an exact no-op. SIByL u_pMin/u_pMax.
RWStructuredBuffer<uint4>         gVoxelLiveBoundMin : register(u18);
RWStructuredBuffer<uint4>         gVoxelLiveBoundMax : register(u19);

// Fuzzy 4-nearest superpixel blend (SIByL u_fuzzyWeight / u_fuzzyIdx, written
// by the superpixel pass): the 4 nearest superpixel centers per pixel and
// their normalized 1/dist^2 weights. Top-level cluster selection becomes a
// mixture over these parents. Global-heap slots 531/532.
RWTexture2D<float4> gFuzzyWeights : register(u20);
RWTexture2D<int4>   gFuzzyIndices : register(u21);

cbuffer VoxelGridCB : register(b4)
{
    float3 voxGridMin;
    float  voxVoxelSize;
    float3 voxGridMax;
    uint   voxGridDim;
    uint   voxInjectUseAvg;
    uint   _voxReserved0;
    float  voxHeatScale;
    uint   voxReuseGiVpl; // ADR 0009: 1 = this raygen's BSDF subtree writes the VPL data
}

float UnpackIrradiance(uint packed)
{
    return float(packed) / 100.0f;
}

// Fixed-point irradiance packing (matches SIByL VXPG: scalar = 100)
uint PackIrradiance(float unpacked)
{
    return uint(unpacked * 100.0f);
}

// ---- Payload (carries first-hit position for guide pdf evaluation) ----

// Minimal hit-ID payload (ADR 0007): the closest hit reports WHAT was hit,
// raygen reconstructs the surface and shades in the bounce loop. No RNG seed
// crosses the payload — no shader outside raygen consumes randoms.
struct GuidedPayload
{
    uint   instanceId;
    uint   primitiveId;
    float2 barycentrics;
    uint   hitFlag;  // 1 = ray hit geometry
};

// ---- BSDF pdf evaluation (mixture of GGX + cosine, matches sampling) ----

float PdfGGX(SurfaceData s, float3 dir)
{
    float3 H = normalize(s.V + dir);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);
    float D = DistributionGGX(NdotH, s.roughness);
    return D * NdotH / (4.0 * VdotH);
}

float PdfCosine(SurfaceData s, float3 dir)
{
    return max(dot(s.N, dir), 0.0) / PI;
}

float PdfBsdf(SurfaceData s, float specularProb, float3 dir)
{
    if (dot(dir, s.N) <= 0.0)
        return 0.0;
    return specularProb * PdfGGX(s, dir) + (1.0 - specularProb) * PdfCosine(s, dir);
}

// Full BRDF eval consistent with the vanilla bounce estimator: GGX specular
// with the path-tracing Smith G (k = a^2/2, NOT the direct-lighting remap)
// plus Lambertian diffuse weighted by kD at NdotV — matches vanilla Hit()
// so the MIS estimator converges to the same image.
float3 EvalBsdfBounce(SurfaceData s, float3 dir)
{
    float NdotL = dot(s.N, dir);
    if (NdotL <= 0.0)
        return float3(0, 0, 0);
    NdotL = max(NdotL, EPSILON);

    float3 H = normalize(s.V + dir);
    float NdotH = max(dot(s.N, H), EPSILON);
    float VdotH = max(dot(s.V, H), EPSILON);

    float  D = DistributionGGX(NdotH, s.roughness);
    float  G = SmithG_GGX(s.NdotV, NdotL, s.roughness);
    float3 F = FresnelSchlick(VdotH, s.F0);
    float3 specular = (D * G * F) / (4.0 * s.NdotV * NdotL + EPSILON);

    // kD from Fresnel at NdotV — same as vanilla's path selection weighting
    float3 Fn = FresnelSchlick(s.NdotV, s.F0);
    float3 kD = (1.0 - Fn) * (1.0 - s.metallic);
    float3 diffuse = kD * s.albedo / PI;

    return specular + diffuse;
}

// selector must come from a stream independent of xi; reusing a hash of xi
// conditions the direction on the lobe choice and biases the mixture sample.
float3 SampleBsdfDir(SurfaceData s, float specularProb, float2 xi, float selector, out float pdf)
{
    float3 dir;
    if (selector < specularProb)
    {
        float3 H = ImportanceSampleGGX(xi, s.N, s.roughness);
        dir = reflect(-s.V, H);
    }
    else
    {
        dir = CosineSampleHemisphere(xi, s.N);
    }
    pdf = PdfBsdf(s, specularProb, dir);
    return dir;
}

// ---- Voxel guide distribution (tree-backed, vxguiding-gi strategy 5) ----

// uint16 leaf ceiling of the bottom tree (Constants::Graphics::LIGHT_TREE_MAX_LEAVES).
#define LIGHT_TREE_MAX_LEAVES 32768

// Lit-voxel count clamped exactly like tree-encode clamped it when building
// this frame's tree — the raw counter would put the leaf boundary past the
// real tree on overflow frames (SIByL reads the raw counter; ADR 0003).
uint LitVoxelCount()
{
    return min(gVoxCounters[0], LIGHT_TREE_MAX_LEAVES);
}

// ---- Voxel solid-angle sampling (SIByL SampleSphericalVoxel family) ----
// The <=3 AABB faces facing the shading point become spherical quads; a face
// is CDF-picked by its exact solid angle, then sampled uniformly within it
// (Urena). The resulting distribution is uniform over the voxel's whole
// silhouette, so pdf = 1 / (total visible solid angle) — direction-independent,
// which keeps the forward sample and the reverse query trivially consistent.
// Every drawn direction geometrically enters the voxel: the semi-NEE gate can
// only fail on occlusion, never on a geometric miss (unlike the old cone).
// Shipped SIByL bakes full-cube per-voxel bounds (tight bounds off, ADR 0004),
// so the plain cube AABB here matches its compact_bound path exactly.

// Permutation matrices mapping world axes onto each face's local frame
// (local z = face normal axis). SIByL rotations[3].
static const float3x3 kVoxelFaceRotations[3] = {
    float3x3(+0, +1, +0, +0, +0, +1, +1, +0, +0), // x axis rotation
    float3x3(+1, +0, +0, +0, +0, +1, +0, +1, +0), // y axis rotation
    float3x3(+1, +0, +0, +0, +1, +0, +0, +0, +1), // z axis rotation
};

// Builds the three candidate face quads of voxel v as seen from shadingPos and
// returns each face's solid angle (0 for edge-on / degenerate faces). Shared
// by the forward sampler and the reverse pdf so both see identical geometry.
// Sampling targets the voxel's COMPACT geometry bounds (SIByL UnpackCompactAABB
// on u_pMin/u_pMax); the semi-NEE gate aabb stays the FULL cube — SIByL
// semantics (gi.slang gates on the VoxelToBound out param). With full-cube
// bakes (useCompact off) compact == cube and behavior is unchanged.
void BuildVoxelFaceQuads(
    float3 shadingPos, int3 v,
    out SphericalQuad squads[3], out float3 locals[3], out float3 faceSolidAngles,
    out float3 aabbMin, out float3 aabbMax)
{
    aabbMin = voxGridMin + float3(v) * voxVoxelSize;
    aabbMax = aabbMin + voxVoxelSize;

    // SIByL UnpackCompactAABB: quantized uint bounds -> world, relative to the cube.
    const uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
    const float3 compactMin = float3(gVoxelLiveBoundMin[flatId].xyz) / 4294967295.0f * voxVoxelSize + aabbMin;
    const float3 compactMax = float3(gVoxelLiveBoundMax[flatId].xyz) / 4294967295.0f * voxVoxelSize + aabbMin;

    const float3 center = 0.5f * (compactMin + compactMax);
    const float3 extend = 0.5f * (compactMax - compactMin);

    // sign() = 0 exactly on a face plane => that face contributes nothing.
    const float3 dirSign = sign(shadingPos - center);
    const float3 xFaceCenter = center + float3(dirSign.x * extend.x, 0, 0);
    const float3 yFaceCenter = center + float3(0, dirSign.y * extend.y, 0);
    const float3 zFaceCenter = center + float3(0, 0, dirSign.z * extend.z);

    locals[0] = mul(kVoxelFaceRotations[0] * dirSign.x, shadingPos - xFaceCenter);
    squads[0] = CreateSphericalQuad(locals[0], extend.yz);
    locals[1] = mul(kVoxelFaceRotations[1] * dirSign.y, shadingPos - yFaceCenter);
    squads[1] = CreateSphericalQuad(locals[1], extend.xz);
    locals[2] = mul(kVoxelFaceRotations[2] * dirSign.z, shadingPos - zFaceCenter);
    squads[2] = CreateSphericalQuad(locals[2], extend.xy);

    faceSolidAngles.x = (dirSign.x == 0) || isnan(squads[0].S) ? 0 : squads[0].S;
    faceSolidAngles.y = (dirSign.y == 0) || isnan(squads[1].S) ? 0 : squads[1].S;
    faceSolidAngles.z = (dirSign.z == 0) || isnan(squads[2].S) ? 0 : squads[2].S;
}

// Reverse-query pdf of the voxel solid-angle distribution: 1 / total visible
// solid angle. SIByL PdfSampleSphericalVoxel — kept verbatim including the
// quirk that a degenerate view (all faces edge-on / shading point effectively
// inside the voxel) returns +inf, which zeroes the BSDF sample's MIS weight.
float PdfVoxelSolidAngle(float3 shadingPos, int3 v)
{
    SphericalQuad squads[3];
    float3 locals[3];
    float3 faceSolidAngles;
    float3 aabbMin, aabbMax;
    BuildVoxelFaceQuads(shadingPos, v, squads, locals, faceSolidAngles, aabbMin, aabbMax);
    return 1.0f / (faceSolidAngles.x + faceSolidAngles.y + faceSolidAngles.z);
}

// Walks a superpixel's 64-slot implicit importance heap from the root down to
// one of the 32 cluster leaves (5 binary decisions by child-sum ratio).
// Returns the cluster index and its selection pdf, or -1 when the heap root is
// 0 (guide dead for this superpixel). One random draw drives the whole walk:
// after every branch the surviving interval is re-stretched to [0,1).
// SIByL SampleTopLevelTree (vxguiding_interface.hlsli).
int SampleSuperpixelClusterHeap(uint spixelFlat, float rnd, out GuidePdf pdfTop)
{
    const uint heapBase = spixelFlat * 64u;
    pdfTop = 0.0;
    if (gSpixelClusterImportanceHeap[heapBase + 1u] == 0.0f)
        return -1;

    uint heapIndex = 1u;
    GuidePdf walkPdf = 1.0;
    [unroll] for (uint level = 0u; level < 5u; ++level)
    {
        const uint leftIndex = heapIndex << 1;
        const float leftImportance  = gSpixelClusterImportanceHeap[heapBase + leftIndex];
        const float rightImportance = gSpixelClusterImportanceHeap[heapBase + leftIndex + 1u];
        float probLeft;
        if (leftImportance == 0.0f)       probLeft = 0.0f;
        else if (rightImportance == 0.0f) probLeft = 1.0f;
        else probLeft = leftImportance / (leftImportance + rightImportance);

        if (rnd < probLeft)
        {
            heapIndex = leftIndex;
            rnd /= probLeft;
            walkPdf *= GuidePdf(probLeft);
        }
        else
        {
            heapIndex = leftIndex + 1u;
            rnd = (rnd - probLeft) / (1.0f - probLeft);
            walkPdf *= GuidePdf(1.0f - probLeft);
        }
    }
    pdfTop = walkPdf;
    return int(heapIndex - 32u);
}

// ---- SLC geometry/distance tree weighting (SIByL slc_common.hlsli) ----------
// Enabled by vxpg.tree.weightMode (guidingFlags bits 5-6, decoded per pixel as
// treeWeightMode): 0 = intensity-only, 1 = geometry bound + avg-minmax distance
// (the paper's distanceType==2). All pure functions over a child AABB and the
// shading point — direct ports; f16-unpacked node bounds are far coarser than
// the geometry error here so the half precision is harmless.

// Max signed distance from p to the AABB projected on dir (SIByL MaxDistAlong).
float MaxDistAlong(float3 p, float3 dir, float3 bmin, float3 bmax)
{
    const float3 dirP = dir * p;
    const float3 lo = dir * bmin - dirP;
    const float3 hi = dir * bmax - dirP;
    return max(lo.x, hi.x) + max(lo.y, hi.y) + max(lo.z, hi.z);
}

// Any orthonormal tangent frame of v1 (SIByL CoordinateSystem_).
void BuildTangentFrame(float3 v1, out float3 v2, out float3 v3)
{
    if (abs(v1.x) > abs(v1.y)) v2 = float3(-v1.z, 0, v1.x) * rsqrt(v1.x * v1.x + v1.z * v1.z);
    else                       v2 = float3(0, v1.z, -v1.y) * rsqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = normalize(cross(v1, v2));
}

// Min |signed distance| from p to the AABB along dir; 0 if the box straddles p
// on dir (SIByL AbsMinDistAlong).
float AbsMinDistAlong(float3 p, float3 dir, float3 bmin, float3 bmax)
{
    const float a = dot(dir, float3(bmin.x, bmin.y, bmin.z) - p);
    const float b = dot(dir, float3(bmin.x, bmin.y, bmax.z) - p);
    const float c = dot(dir, float3(bmin.x, bmax.y, bmin.z) - p);
    const float d = dot(dir, float3(bmin.x, bmax.y, bmax.z) - p);
    const float e = dot(dir, float3(bmax.x, bmin.y, bmin.z) - p);
    const float f = dot(dir, float3(bmax.x, bmin.y, bmax.z) - p);
    const float g = dot(dir, float3(bmax.x, bmax.y, bmin.z) - p);
    const float h = dot(dir, float3(bmax.x, bmax.y, bmax.z) - p);
    const bool hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
    const bool hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
    if (hasPositive && hasNegative) return 0.0f;
    return min(min(min(abs(a), abs(b)), min(abs(c), abs(d))),
               min(min(abs(e), abs(f)), min(abs(g), abs(h))));
}

// Upper bound on the cosine-weighted solid angle of the AABB at (p, n) — the
// SLC "geometry term" (SIByL GeomTermBound). 0 when the box is fully below the
// horizon of n.
float GeomTermBound(float3 p, float3 n, float3 bmin, float3 bmax)
{
    const float nrmMax = MaxDistAlong(p, n, bmin, bmax);
    if (nrmMax <= 0.0f) return 0.0f;
    float3 t, b;
    BuildTangentFrame(n, t, b);
    const float yMin = AbsMinDistAlong(p, t, bmin, bmax);
    const float zMin = AbsMinDistAlong(p, b, bmin, bmax);
    const float hyp2 = yMin * yMin + zMin * zMin + nrmMax * nrmMax;
    return nrmMax * rsqrt(hyp2);
}

// Cheap variant (SIByL GeomTermBoundApproximate): drops the tangent frame and the
// two 8-corner AbsMinDistAlong passes, using the single closest-point tangential
// offset instead. ~1/5 the per-node cost of GeomTermBound; for voxel-sized boxes
// far from p the bound is nearly identical. Used by treeWeightMode == 2.
float GeomTermBoundApproximate(float3 p, float3 n, float3 bmin, float3 bmax)
{
    const float nrmMax = MaxDistAlong(p, n, bmin, bmax);
    if (nrmMax <= 0.0f) return 0.0f;
    const float3 d = min(max(p, bmin), bmax) - p; // p -> closest point on the AABB
    const float3 tng = d - dot(d, n) * n;         // tangential component vs n
    const float hyp2 = dot(tng, tng) + nrmMax * nrmMax;
    return nrmMax * rsqrt(hyp2);
}

float SquaredDistToClosest(float3 p, float3 bmin, float3 bmax)
{
    const float3 d = min(max(p, bmin), bmax) - p;
    return dot(d, d);
}

float SquaredDistToFarthest(float3 p, float3 bmin, float3 bmax)
{
    const float3 d = max(abs(bmin - p), abs(bmax - p));
    return dot(d, d);
}

// SIByL normalizedWeights: the l2 terms are SWAPPED so a nearer child (smaller
// l2) gets the larger share. Guarded against a 0/0 (both intensGeom 0).
float NormalizedWeights(float l2_0, float l2_1, float ig0, float ig1)
{
    const float ww0 = l2_1 * ig0;
    const float ww1 = l2_0 * ig1;
    const float denom = ww0 + ww1;
    return (denom > 0.0f) ? ww0 / denom : 0.5f;
}

// Probability of descending to the LEFT child. Returns false only on a fully
// dead branch (both children unreachable) — the caller drops the guided sample.
// treeWeightMode 0 = intensity ratio (telescopes on the reverse side);
// 1 = SLC geometry (exact GeomTermBound) + avg-minmax distance (paper);
// 2 = same but the CHEAP GeomTermBoundApproximate (lower per-node cost).
// SIByL EvaluateFirstChildWeight.
bool FirstChildProb(LightTreeNode c0, LightTreeNode c1, float3 p, float3 n,
                    uint treeWeightMode, out float prob0)
{
    prob0 = 0.0f;
    const float i0 = c0.intensity;
    const float i1 = c1.intensity;

    if (treeWeightMode == 0u)
    {
        if (i0 == 0.0f) { if (i1 == 0.0f) return false; prob0 = 0.0f; return true; }
        if (i1 == 0.0f) { prob0 = 1.0f; return true; }
        prob0 = i0 / (i0 + i1);
        return true;
    }

    const float3 c0Min = UnpackFloat3(c0.aabbMin);
    const float3 c0Max = UnpackFloat3(c0.aabbMax);
    const float3 c1Min = UnpackFloat3(c1.aabbMin);
    const float3 c1Max = UnpackFloat3(c1.aabbMax);

    float geom0, geom1;
    if (treeWeightMode == 2u)
    {
        geom0 = GeomTermBoundApproximate(p, n, c0Min, c0Max);
        geom1 = GeomTermBoundApproximate(p, n, c1Min, c1Max);
    }
    else
    {
        geom0 = GeomTermBound(p, n, c0Min, c0Max);
        geom1 = GeomTermBound(p, n, c1Min, c1Max);
    }

    if (geom0 + geom1 == 0.0f) return false;
    if (geom0 == 0.0f) { prob0 = 0.0f; return true; }
    if (geom1 == 0.0f) { prob0 = 1.0f; return true; }

    const float ig0 = i0 * geom0;
    const float ig1 = i1 * geom1;

    const float l2min0 = SquaredDistToClosest(p, c0Min, c0Max);
    const float l2min1 = SquaredDistToClosest(p, c1Min, c1Max);
    const float l2max0 = SquaredDistToFarthest(p, c0Min, c0Max);
    const float l2max1 = SquaredDistToFarthest(p, c1Min, c1Max);

    // avg of the closest-point and farthest-point weightings (paper).
    const float igSum = ig0 + ig1;
    const float wMax0 = (l2min0 == 0.0f && l2min1 == 0.0f)
        ? ((igSum > 0.0f) ? ig0 / igSum : 0.5f)
        : NormalizedWeights(l2min0, l2min1, ig0, ig1);
    const float wMin0 = NormalizedWeights(l2max0, l2max1, ig0, ig1);
    prob0 = 0.5f * (wMax0 + wMin0);
    return true;
}

// Descends the bottom light tree from a cluster root to a leaf. Node ids >=
// leafStartIndex (= leafCount - 1) are leaves. Returns the leaf NODE id (not
// compactID), or -1 on a dead branch (both children unreachable at this shading
// point — more common under geometry weighting; the guided sample is then
// dropped and BSDF MIS carries full weight). SIByL TraverseLightTree.
int TraverseLightTreeToLeaf(int nodeId, uint leafStartIndex, float rnd,
                            float3 p, float3 n, uint treeWeightMode, out GuidePdf pdfTree)
{
    pdfTree = 1.0;
    [loop] while (uint(nodeId) < leafStartIndex)
    {
        const uint leftChild  = uint(gLightTreeNodes[nodeId].leftIndex);
        const uint rightChild = uint(gLightTreeNodes[nodeId].rightIndex);

        float probLeft;
        if (!FirstChildProb(gLightTreeNodes[leftChild], gLightTreeNodes[rightChild],
                            p, n, treeWeightMode, probLeft))
            return -1; // dead branch: invalid sample

        if (rnd < probLeft)
        {
            nodeId = int(leftChild);
            rnd /= probLeft;
            pdfTree *= GuidePdf(probLeft);
        }
        else
        {
            nodeId = int(rightChild);
            rnd = (rnd - probLeft) / (1.0f - probLeft);
            pdfTree *= GuidePdf(1.0f - probLeft);
        }
    }
    return nodeId;
}

// Reverse pdf of the tree walk. Intensity-ratio branching telescopes — every
// intermediate intensity cancels — leaving one division. SIByL
// PdfTraverseLightTree_Intensity.
GuidePdf PdfTraverseLightTreeIntensity(int clusterRootId, int leafNodeId)
{
    return GuidePdf(gLightTreeNodes[leafNodeId].intensity) /
           GuidePdf(gLightTreeNodes[clusterRootId].intensity);
}

// Reverse pdf, general: geometry weighting does NOT telescope, so re-evaluate
// the split probability at every parent from the leaf up to the cluster root,
// multiplying the branch the path actually took. Re-uses FirstChildProb so the
// value matches the forward walk's pdfTree exactly (consistent estimator).
// SIByL PdfTraverseLightTree.
GuidePdf PdfTraverseLightTree(int clusterRootId, int leafNodeId, float3 p, float3 n, uint treeWeightMode)
{
    if (treeWeightMode == 0u)
        return PdfTraverseLightTreeIntensity(clusterRootId, leafNodeId);

    GuidePdf pdf = 1.0;
    int nodeId = leafNodeId;
    if (nodeId == clusterRootId)
        return pdf;

    [loop] while (true)
    {
        const uint parentId = uint(gLightTreeNodes[nodeId].parentIndex);
        if (parentId == 0xFFFFu)
            break; // reached the whole tree's root
        const uint leftChild  = uint(gLightTreeNodes[parentId].leftIndex);
        const uint rightChild = uint(gLightTreeNodes[parentId].rightIndex);

        float probLeft;
        // Ignore the dead-branch bool: on a path the forward walk reached this
        // cannot fire, and if it did FirstChildProb leaves probLeft = 0, which
        // zeroes the pdf (BSDF MIS then takes full weight) — SIByL's behavior.
        FirstChildProb(gLightTreeNodes[leftChild], gLightTreeNodes[rightChild],
                       p, n, treeWeightMode, probLeft);
        pdf *= (uint(nodeId) == leftChild) ? GuidePdf(probLeft) : GuidePdf(1.0 - probLeft);

        nodeId = int(parentId);
        if (nodeId == clusterRootId)
            break;
    }
    return pdf;
}

// Mixture probability of the fuzzy parent set picking `cluster`: sum over the
// (renormalized) parents of weight x heapLeaf/heapRoot — SIByL strategy 7's
// reverse formula (gi.slang:389-403), used here on BOTH the forward and the
// reverse side. The shipped strategy 6 pairs a single-parent forward pdf with
// this mixture reverse — an inconsistent estimator; consistent-mixture is a
// deliberate deviation (ADR 0003).
GuidePdf FuzzyClusterMixturePdf(float4 fuzzyWeights, int4 fuzzyIndices, int cluster)
{
    GuidePdf pdfTop = 0.0;
    [unroll] for (int f = 0; f < 4; ++f)
    {
        if (fuzzyWeights[f] > 0.0 && fuzzyIndices[f] >= 0)
        {
            const uint heapBase = uint(fuzzyIndices[f]) * 64u;
            const float rootImportance = gSpixelClusterImportanceHeap[heapBase + 1u];
            const float leafImportance = gSpixelClusterImportanceHeap[heapBase + 32u + uint(cluster)];
            if (rootImportance > 0.0f)
                pdfTop += GuidePdf(fuzzyWeights[f]) * GuidePdf(leafImportance) / GuidePdf(rootImportance);
        }
    }
    return pdfTop;
}

// Guide pdf of a BSDF-sampled ray evaluated at its hit position: the mixture
// probability the fuzzy parent set picks the hit voxel's cluster, times the
// telescoped tree leaf/root, times the direction pdf toward that voxel. 0
// whenever any link is missing — hit voxel unlit, cluster unassigned (such
// leaves sort past every cluster run and sit under no cluster root, so the
// forward walk truly cannot reach them), or every parent heap empty — which
// hands the BSDF sample full MIS weight.
// (vxguiding-gi.slang strategy 5 reverse block + strategy 7 fuzzy top pdf.)
GuidePdf EvalTreeGuidePdf(float4 fuzzyWeights, int4 fuzzyIndices, float3 shadingPos,
                          float3 shadingNormal, float3 hitPos, uint treeWeightMode)
{
    int3 v = int3(floor((hitPos - voxGridMin) / voxVoxelSize));
    if (any(v < 0) || any(v >= int(voxGridDim)))
        return 0.0;

    const uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;

    const int compactID = gVoxInverseIndex[flatId];
    if (compactID < 0 || uint(compactID) >= LitVoxelCount())
        return 0.0;

    const int cluster = gVoxelClusterAssignments[compactID];
    if (cluster < 0 || cluster >= 32)
        return 0.0;

    const GuidePdf pdfTop = FuzzyClusterMixturePdf(fuzzyWeights, fuzzyIndices, cluster);
    if (pdfTop == 0.0)
        return 0.0;

    const int leafNodeId    = gCompactToLeaf[compactID];
    const int clusterRootId = gClusterRootNodes[cluster];
    if (leafNodeId < 0 || clusterRootId < 0)
        return 0.0;
    const GuidePdf pdfTree = PdfTraverseLightTree(clusterRootId, leafNodeId, shadingPos, shadingNormal, treeWeightMode);

    const float pdfDir = PdfVoxelSolidAngle(shadingPos, v);
    return pdfTop * pdfTree * GuidePdf(pdfDir);
}

// Decodes a compact-list flat voxel id back to grid coordinates.
int3 VoxelCoordFromFlatId(uint flatId)
{
    int3 v;
    v.x = int(flatId % voxGridDim);
    v.y = int((flatId / voxGridDim) % voxGridDim);
    v.z = int(flatId / (voxGridDim * voxGridDim));
    return v;
}

// Samples a direction uniformly over voxel v's visible solid angle (CDF over
// the face solid angles, then a uniform spherical-quad draw within the picked
// face) and outputs the voxel AABB for the semi-NEE gate. xi.x picks the face,
// xi.yz sample the quad. pdf = 1 / total visible solid angle; 0 with a zero
// direction when the view is degenerate (SIByL returns NaN there — behavior is
// identical downstream because the caller gates on pdf > 0 before tracing).
// SIByL SampleSphericalVoxel (vxguiding_interface.hlsli, float3-extend).
float3 SampleVoxelSolidAngle(
    float3 shadingPos, int3 v, float3 xi,
    out float pdfDir, out float3 aabbMin, out float3 aabbMax)
{
    SphericalQuad squads[3];
    float3 locals[3];
    float3 faceSolidAngles;
    BuildVoxelFaceQuads(shadingPos, v, squads, locals, faceSolidAngles, aabbMin, aabbMax);

    // sign() of the shading point relative to the voxel center, rebuilt for
    // the local->world transform of the sampled direction.
    const float3 center = 0.5f * (aabbMin + aabbMax);
    const float3 dirSign = sign(shadingPos - center);

    float cdfs[3];
    float sum = 0.0f;
    pdfDir = 0.0f;
    [unroll] for (int i = 0; i < 3; ++i)
    {
        if (faceSolidAngles[i] > 0.0f)
            sum += faceSolidAngles[i];
        cdfs[i] = sum;
    }
    if (sum == 0.0f)
        return float3(0, 0, 0);
    xi.x *= sum;

    int selectedFace = -1;
    float facePickPdf = 0.0f;
    [unroll] for (int j = 0; j < 3; ++j)
    {
        if (xi.x <= cdfs[j])
        {
            selectedFace = j;
            facePickPdf = faceSolidAngles[j] / sum;
            break;
        }
    }

    float3 localDir;
    float faceQuadPdf;
    SampleSphericalQuad(locals[selectedFace], squads[selectedFace], xi.yz, localDir, faceQuadPdf);

    pdfDir = facePickPdf * faceQuadPdf; // telescopes to 1 / sum
    return mul(localDir, kVoxelFaceRotations[selectedFace] * dirSign[selectedFace]);
}

// ---- MIS weight (balance or power heuristic via guidingFlags bit 0) ----

float MisWeight(float wSelf, float wOther)
{
    if ((guidingFlags & 1u) != 0u)
    {
        wSelf  *= wSelf;
        wOther *= wOther;
    }
    // No epsilon in the denominator (SIByL w1/(w1+w2) verbatim): both call
    // sites gate on pdf > 0 and isnan-guard the other strategy's pdf, so the
    // denominator is strictly positive. An epsilon here makes the two MIS
    // weights sum below 1 — an energy tax that concentrates exactly where
    // squared pdfs are small (grazing/penumbra directions; measured as the
    // 0.026-FLIP darkening on Deep Light before removal).
    return wSelf / (wSelf + wOther);
}

// ---- Continuation trace (deeper bounces, vanilla logic) ----

// Sky reached by an indirect ray: sky-lighting switch + firefly clamp applied
// unconditionally (every loop segment is indirect; primary sky is sampled in
// raygen). Identical to raytracing.hlsl's Miss so both techniques converge to
// the same target. Lived in GuidedMiss before ADR 0007 moved shading to raygen.
float3 IndirectSkyRadiance(float3 dir)
{
    if (skyLightingEnabled == 0)
        return float3(0, 0, 0);
    float u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = -asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    float3 sky = g_skybox.SampleLevel(gsamLinearWrap, float2(u, v), 0).rgb;
    if (indirectSkyClamp > 0.0)
        sky = min(sky, indirectSkyClamp.xxx);
    return sky;
}

// ---- Bounce-trace backend (pipeline TraceRay vs inline RayQuery) ----

#ifdef GUIDED_TRACE_RQ

// Inline-RayQuery backend (compute integrator): the GuidedAnyHit alpha test
// replays in the candidate loop; the committed hit fills the same minimal
// hit-ID payload the closest-hit shader reports in the pipeline path.
GuidedPayload TraceBounceRay(RayDesc ray)
{
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(SceneBVH, RAY_FLAG_NONE, ~0, ray);
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            InstanceInfo instance = g_instanceInfo[query.CandidateInstanceID()];
            uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
            uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
            HitData hit = GetHitData(query.CandidatePrimitiveIndex(), vertexOffset, indexOffset,
                                     query.CandidateTriangleBarycentrics(), instance.objectToWorld);
            float4 albedo = SampleTextureColor(instance, hit) * instance.baseColorFactor;
            if (albedo.a >= EPSILON)
                query.CommitNonOpaqueTriangleHit();
        }
    }

    GuidedPayload p;
    p.instanceId = 0;
    p.primitiveId = 0;
    p.barycentrics = float2(0, 0);
    p.hitFlag = 0;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        p.instanceId = query.CommittedInstanceID();
        p.primitiveId = query.CommittedPrimitiveIndex();
        p.barycentrics = query.CommittedTriangleBarycentrics();
        p.hitFlag = 1;
    }
    return p;
}

#else

GuidedPayload TraceBounceRay(RayDesc ray)
{
    GuidedPayload p;
    p.instanceId = 0;
    p.primitiveId = 0;
    p.barycentrics = float2(0, 0);
    p.hitFlag = 0;
    TraceRay(SceneBVH, 0, ~0, 0, 1, 0, ray, p);
    return p;
}

#endif // GUIDED_TRACE_RQ

// Launch coordinates shared by both entry points (raygen intrinsics do not
// exist in compute; a per-thread static avoids threading them through every
// helper).
static uint2 gLaunchIndex;
static uint2 gLaunchDims;

// VPL injection from the BSDF subtree's first bounce vertex (ADR 0009): the
// same writes the dedicated injection trace performs — per-pixel VPL position,
// per-voxel representative (last-writer-wins), and the packed-irradiance
// atomics. Position/normal write on any hit; irradiance only when the vertex
// receives direct light (mirrors lightInjection.hlsl exactly).
void InjectVplFromBounce(float3 position, float3 shadingNormal, float3 directLight)
{
    const float packedNormal = asfloat(UnitVectorToUnorm32Octahedron(shadingNormal));
    gVplPosition[gLaunchIndex] = float4(position, packedNormal);

    int3 voxelIdx = int3(floor((position - voxGridMin) / voxVoxelSize));
    if (any(voxelIdx < 0) || any(voxelIdx >= int(voxGridDim)))
        return;

    gVoxelRepresentative[voxelIdx] = float4(position, packedNormal);

    float irradiance = max(directLight.r, max(directLight.g, directLight.b));
    if (irradiance <= 0.0)
        return;

    uint packedIrr = PackIrradiance(irradiance);
    if (voxInjectUseAvg != 0)
    {
        uint old;
        InterlockedAdd(gVoxIrradiance[voxelIdx], packedIrr, old);
        InterlockedAdd(gVoxVplCount[voxelIdx], 1u, old);
    }
    else
    {
        uint old;
        InterlockedMax(gVoxIrradiance[voxelIdx], packedIrr, old);
        gVoxVplCount[voxelIdx] = 1u;
    }
}

// Flat iterative bounce loop (ADR 0007): replaces the recursive closest-hit
// continuation with the same estimator — per vertex add throughput-weighted
// direct light, sample the next bounce, stop at numBounces. All rays (bounce +
// shadow) launch from raygen; the closest hit only reports hit IDs and the
// surface is reconstructed here through the same instance-parameterized path
// the VBuffer first vertex uses.
// writeVpl: only the BSDF MIS subtree passes true (ADR 0009) — fitting the
// guide from guided samples would be a self-reinforcing feedback loop.
float3 TraceIndirect(float3 origin, float3 dir, inout uint seed, bool writeVpl, out float3 hitPos, out bool didHit)
{
    float3 radiance = float3(0, 0, 0);
    float3 pathThroughput = float3(1, 1, 1);
    float3 rayOrigin = origin;
    float3 rayDir = dir;
    hitPos = float3(0, 0, 0);
    didHit = false;

    for (uint bounce = 1; bounce <= (uint)numBounces; ++bounce)
    {
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin = RAY_TMIN;
        ray.TMax = RAY_TMAX;

        GuidedPayload p = TraceBounceRay(ray);

        if (p.hitFlag == 0u)
        {
            radiance += pathThroughput * IndirectSkyRadiance(rayDir);
            break;
        }

        InstanceInfo instance = g_instanceInfo[p.instanceId];
        GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
        HitData hit = GetHitData(p.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                                 p.barycentrics, instance.objectToWorld);

        if (bounce == 1u)
        {
            hitPos = hit.position;  // first hit: the semi-NEE gate checks this
            didHit = true;
        }

        float3 albedo = SampleTextureColor(instance, hit).rgb * instance.baseColorFactor.rgb;
        float2 rm = SampleRoughnessMetallic(instance, hit);
        float roughness = max(rm.x, MIN_ROUGHNESS);
        float metallic = rm.y;

        float3 N = SampleWorldSpaceNormal(instance, hit);
        float3 V = -rayDir;

        float3 geometricN = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
        if (dot(geometricN, V) < 0.0)
            N = -N;

        SurfaceData surface;
        surface.N         = N;
        surface.V         = V;
        surface.NdotV     = max(dot(N, V), 1e-4);  // div-by-zero guard only; 0.1 floored grazing specular
        surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
        surface.albedo    = albedo;
        surface.roughness = roughness;
        surface.metallic  = metallic;

        const float3 directLight = CalculateDirectLightning(hit, surface);
        radiance += pathThroughput * directLight;

        // ADR 0009: the first bounce vertex of the BSDF subtree doubles as
        // next frame's injection sample.
        if (bounce == 1u && writeVpl && voxReuseGiVpl != 0u)
            InjectVplFromBounce(hit.position, N, directLight);

        if (bounce >= (uint)numBounces)
            break;

        // Continuation sample: verbatim port of the old closest-hit logic
        // (specular/diffuse selection, pdf-cancelled throughput).
        float3 F = FresnelSchlick(surface.NdotV, surface.F0);
        float specularProb = (F.r + F.g + F.b) / 3.0;

        float2 xi = Random2D(seed);
        seed = pcg_hash(seed);

        // Independent lobe selector (see SampleBsdfDir): xi-hash reuse biases.
        float pathSelector = Random1D(seed);
        seed = pcg_hash(seed);

        float3 bounceDir;
        float3 throughput;

        if (pathSelector < specularProb)
        {
            float3 H = ImportanceSampleGGX(xi, N, roughness);
            bounceDir = reflect(-V, H);
            throughput = EvalSpecularBounce(surface, H, bounceDir);
            if (all(throughput == 0))
                break; // invalid bounce sample — direct light at this vertex stands
            throughput /= specularProb;
        }
        else
        {
            float3 kD = (1.0 - F) * (1.0 - metallic);
            bounceDir = CosineSampleHemisphere(xi, N);
            throughput = EvalDiffuseBounce(surface, kD, bounceDir);
            if (all(throughput == 0))
                break; // invalid bounce sample — direct light at this vertex stands
            throughput /= (1.0 - specularProb);  // branch guarantees pathSelector>=specularProb => 1-specularProb>0
        }

        pathThroughput *= throughput;
        rayOrigin = hit.position;
        rayDir = bounceDir;
    }

    return radiance;
}

// ---- Second-bounce guiding (SIByL strategy-6 `second=true`) -----------------
// SIByL guides the SECOND path vertex too, via the GLOBAL irradiance guide
// (VG_Irradiance: traverse the light tree from the root node 0 by intensity, no
// superpixel/cluster/fuzzy top level). Enabled by vxpg.secondBounce
// (guidingFlags bit 7). This turns the estimator into a 2-bounce guided path
// (vertex1 MIS -> NEE at vertex2 -> vertex2 MIS -> NEE at vertex3), so run the
// A/B at bounces == 2. Deviation from Bamboo's flat N-bounce tail (ADR 0007),
// gated off by default.

// One indirect segment: trace a ray, return terminal NEE at the hit (SIByL
// EvaluateIndirectLight) or the sky. hitPos/didHit feed the guide semi-NEE gate
// and the reverse pdf.
float3 TraceOneBounceNEE(float3 origin, float3 dir, inout uint seed, out float3 hitPos, out bool didHit)
{
    hitPos = float3(0, 0, 0);
    didHit = false;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    GuidedPayload p = TraceBounceRay(ray);
    if (p.hitFlag == 0u)
        return IndirectSkyRadiance(dir);

    InstanceInfo instance = g_instanceInfo[p.instanceId];
    GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
    HitData hit = GetHitData(p.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                             p.barycentrics, instance.objectToWorld);

    float3 albedo = SampleTextureColor(instance, hit).rgb * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(instance, hit);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;

    float3 N = SampleWorldSpaceNormal(instance, hit);
    float3 V = -dir;
    float3 geometricN = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
    if (dot(geometricN, V) < 0.0)
        N = -N;

    SurfaceData surface;
    surface.N         = N;
    surface.V         = V;
    surface.NdotV     = max(dot(N, V), 1e-4);
    surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
    surface.albedo    = albedo;
    surface.roughness = roughness;
    surface.metallic  = metallic;

    hitPos = hit.position;
    didHit = true;
    return CalculateDirectLightning(hit, surface);
}

// Reverse pdf of the GLOBAL irradiance guide at a BSDF sample's hit voxel:
// P(select the hit voxel from tree root 0, intensity-only) x P(dir | voxel).
// 0 for sky / unlit / out-of-tree hits -> BSDF sample takes full MIS weight.
GuidePdf EvalGlobalGuidePdf(float3 shadingPos, float3 shadingNormal, float3 hitPos)
{
    int3 v = int3(floor((hitPos - voxGridMin) / voxVoxelSize));
    if (any(v < 0) || any(v >= int(voxGridDim)))
        return 0.0;
    const uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
    const int compactID = gVoxInverseIndex[flatId];
    if (compactID < 0 || uint(compactID) >= LitVoxelCount())
        return 0.0;
    const int leafNodeId = gCompactToLeaf[compactID];
    if (leafNodeId < 0)
        return 0.0;
    const GuidePdf pdfTree = PdfTraverseLightTreeIntensity(0, leafNodeId);
    const float pdfDir = PdfVoxelSolidAngle(shadingPos, v);
    return pdfTree * GuidePdf(pdfDir);
}

// Two-sample MIS (BSDF + global irradiance guide) at the SECOND vertex; returns
// the terminal indirect radiance from that vertex (NEE at the third vertex).
// Mirrors ShadeFirstVertex but with the global guide and a single terminal
// segment (SIByL secondbounce block, gi.slang:445-503).
float3 ShadeSecondVertex(HitData hit2, SurfaceData surf2, float specularProb, inout uint seed)
{
    const uint litVoxelCount = LitVoxelCount();
    float3 radiance = float3(0, 0, 0);

    // BSDF strategy
    {
        float2 xi = Random2D(seed); seed = pcg_hash(seed);
        float selector = Random1D(seed); seed = pcg_hash(seed);
        float pdfB;
        float3 dir = SampleBsdfDir(surf2, specularProb, xi, selector, pdfB);
        if (pdfB > EPSILON && dot(dir, surf2.N) > 0.0)
        {
            float3 f = EvalBsdfBounce(surf2, dir);
            if (any(f > 0))
            {
                float3 hp; bool dh;
                float3 incoming = TraceOneBounceNEE(hit2.position, dir, seed, hp, dh);
                float pdfG = (litVoxelCount > 0u && dh)
                    ? float(EvalGlobalGuidePdf(hit2.position, surf2.N, hp)) : 0.0;
                if (isnan(pdfG)) pdfG = 0.0;
                float weight = MisWeight(pdfB, pdfG);
                radiance += f * dot(surf2.N, dir) * incoming * weight / pdfB;
            }
        }
    }

    // Guide strategy (global irradiance tree, root node 0, intensity-only)
    if (litVoxelCount > 0u)
    {
        float2 walkXi = Random2D(seed); seed = pcg_hash(seed);
        float2 quadXi = Random2D(seed); seed = pcg_hash(seed);
        float2 faceXi = Random2D(seed); seed = pcg_hash(seed);

        GuidePdf pdfTree;
        const int leafNodeId = TraverseLightTreeToLeaf(0, litVoxelCount - 1u, walkXi.x,
                                                       hit2.position, surf2.N, 0u, pdfTree);
        if (leafNodeId >= 0)
        {
            const uint compactID = uint(gLightTreeNodes[leafNodeId].voxelIndex);
            const int3 v = VoxelCoordFromFlatId(gVoxCompactIds[compactID]);
            const int3 currVox = int3(floor((hit2.position - voxGridMin) / voxVoxelSize));
            // Skip a degenerate self-voxel pick (SIByL SampleVoxelGuiding guard).
            if (!all(v == currVox))
            {
                float pdfDir; float3 aabbMin, aabbMax;
                float3 dir = SampleVoxelSolidAngle(hit2.position, v, float3(faceXi.x, quadXi),
                                                   pdfDir, aabbMin, aabbMax);
                const GuidePdf pdfG = pdfTree * GuidePdf(pdfDir);
                if (pdfG > 0.0 && dot(dir, surf2.N) > 0.0)
                {
                    float3 f = EvalBsdfBounce(surf2, dir);
                    if (any(f > 0))
                    {
                        float3 hp; bool dh;
                        float3 incoming = TraceOneBounceNEE(hit2.position, dir, seed, hp, dh);
                        bool accepted = dh && all(hp >= aabbMin) && all(hp <= aabbMax);
                        if (accepted)
                        {
                            float pdfBAtDir = PdfBsdf(surf2, specularProb, dir);
                            if (isnan(pdfBAtDir)) pdfBAtDir = 0.0;
                            float weight = MisWeight(float(pdfG), pdfBAtDir);
                            radiance += f * dot(surf2.N, dir) * incoming * weight / float(pdfG);
                        }
                    }
                }
            }
        }
    }

    return radiance;
}

// BSDF-branch continuation of vertex 1 WITH second-bounce guiding: trace to
// vertex 2, add its NEE, then the guided second-vertex MIS (terminal). Replaces
// TraceIndirect for that branch when vxpg.secondBounce is on. hitPos/didHit are
// vertex 2 (the vertex-1 guide semi-NEE gate + reverse pdf still key on it).
float3 TraceIndirectSecondGuide(float3 origin, float3 dir, inout uint seed, bool writeVpl,
                                out float3 hitPos, out bool didHit)
{
    hitPos = float3(0, 0, 0);
    didHit = false;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = RAY_TMIN;
    ray.TMax = RAY_TMAX;

    GuidedPayload p = TraceBounceRay(ray);
    if (p.hitFlag == 0u)
        return IndirectSkyRadiance(dir);

    InstanceInfo instance = g_instanceInfo[p.instanceId];
    GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
    HitData hit = GetHitData(p.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                             p.barycentrics, instance.objectToWorld);

    float3 albedo = SampleTextureColor(instance, hit).rgb * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(instance, hit);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;
    float3 N = SampleWorldSpaceNormal(instance, hit);
    float3 V = -dir;
    float3 geometricN = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
    if (dot(geometricN, V) < 0.0)
        N = -N;

    SurfaceData surface;
    surface.N         = N;
    surface.V         = V;
    surface.NdotV     = max(dot(N, V), 1e-4);
    surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
    surface.albedo    = albedo;
    surface.roughness = roughness;
    surface.metallic  = metallic;

    hitPos = hit.position;
    didHit = true;

    const float3 directLight = CalculateDirectLightning(hit, surface);
    if (writeVpl && voxReuseGiVpl != 0u)
        InjectVplFromBounce(hit.position, N, directLight);

    float3 F = FresnelSchlick(surface.NdotV, surface.F0);
    float specularProb = (F.r + F.g + F.b) / 3.0;

    return directLight + ShadeSecondVertex(hit, surface, specularProb, seed);
}

// ---- First path vertex: two-sample MIS between BSDF and the tree guide ----
// Runs in raygen on the VBuffer-reconstructed hit (ADR 0004); deeper bounces
// continue through the closest hit. debugView = guidingFlags bits 1-4
// (1 = BSDF strategy only, 2 = guide strategy only, 3 = MIS weight
// false-color, 4 = guided sample acceptance green/red/blue).

float3 ShadeFirstVertex(HitData hit, SurfaceData surface, float specularProb, uint debugView,
                        float4 fuzzyWeights, int4 fuzzyIndices, int spixelFlat, inout uint seed)
{
    if (numBounces == 0)
        return CalculateDirectLightning(hit, surface);

    const uint litVoxelCount = LitVoxelCount();
    // Guide-dead pixels (no lit voxels, or every fuzzy parent's heap empty) run
    // BSDF-only at full MIS weight — no uniform fallback (faithful, see header).
    const bool guideAlive = (litVoxelCount > 0u) && any(fuzzyWeights > 0.0);

    // Bottom light-tree branch weighting (vxpg.tree.weightMode, guidingFlags
    // bits 5-6): 0 = intensity-only, 1 = geometry + avg-minmax distance (paper).
    const uint treeWeightMode = (guidingFlags >> 5) & 3u;
    // Second-bounce guiding (vxpg.secondBounce, bit 7): also MIS-guide the second
    // vertex (SIByL `second=true`). Turns the BSDF branch into a 2-bounce guided
    // path — meaningful only at bounces >= 2.
    const bool guideSecondBounce = ((guidingFlags >> 7) & 1u) != 0u && numBounces >= 2u;

    float3 radiance = (debugView >= 3u) ? float3(0, 0, 0)
                                        : CalculateDirectLightning(hit, surface);

    float misWeightB = 0.0;
    float misWeightG = 0.0;
    // Acceptance classification (view 4): 0 = guide dead (blue), 1 = traced
    // but gate-rejected (red), 2 = accepted (green), 3 = heap walk returned no
    // cluster (cyan), 5 = pdf <= 0 (orange), 6 = sampled direction below the
    // horizon (violet), 7 = zero BRDF toward the sample (yellow).
    uint guideOutcome = 0u;

    // BSDF strategy
    if (debugView != 2u && debugView != 4u)
    {
        float2 xi = Random2D(seed);
        seed = pcg_hash(seed);
        float selector = Random1D(seed);
        seed = pcg_hash(seed);

        float pdfB;
        float3 dir = SampleBsdfDir(surface, specularProb, xi, selector, pdfB);
        if (pdfB > EPSILON && dot(dir, surface.N) > 0.0)
        {
            float3 f = EvalBsdfBounce(surface, dir);
            if (any(f > 0))
            {
                float3 hitPos;
                bool didHit;
                float3 incoming = guideSecondBounce
                    ? TraceIndirectSecondGuide(hit.position, dir, seed, /*writeVpl*/ true, hitPos, didHit)
                    : TraceIndirect(hit.position, dir, seed, /*writeVpl*/ true, hitPos, didHit);

                // Guide pdf at the BSDF sample, evaluated through the reverse
                // chain at the ray's hit voxel (0 for misses / unreachable
                // voxels -> weight 1 for this sample).
                float pdfGAtDir = 0.0;
                if (guideAlive && didHit)
                    pdfGAtDir = float(EvalTreeGuidePdf(fuzzyWeights, fuzzyIndices, hit.position, surface.N, hitPos, treeWeightMode));
                if (isnan(pdfGAtDir)) pdfGAtDir = 0.0; // SIByL w2 guard

                float weight = MisWeight(pdfB, pdfGAtDir);
                misWeightB = weight;
                if (debugView != 3u)
                {
                    float NdotL = dot(surface.N, dir);
                    radiance += f * NdotL * incoming * weight / pdfB;
                }
            }
        }
    }

    // Guide strategy (forward chain: fuzzy parent pick -> heap -> cluster root
    // -> tree -> voxel -> solid-angle sample)
    if (debugView != 1u && guideAlive)
    {
        float2 walkXi = Random2D(seed);
        seed = pcg_hash(seed);
        float2 quadXi = Random2D(seed);
        seed = pcg_hash(seed);
        float2 faceParentXi = Random2D(seed);
        seed = pcg_hash(seed);
        const float faceXi = faceParentXi.x;

        // CDF-pick one fuzzy parent (SIByL strategy 6 forward, gi.slang:301-311);
        // a float fall-through keeps the hard SLIC assignment, like SIByL keeps
        // its pre-initialized u_spixelIdx.
        int selectedParent = spixelFlat;
        float parentAccum = 0.0;
        [unroll] for (int f = 0; f < 4; ++f)
        {
            parentAccum += fuzzyWeights[f];
            if (faceParentXi.y < parentAccum)
            {
                selectedParent = fuzzyIndices[f];
                break;
            }
        }
        if (selectedParent < 0)
            return radiance; // no live parent and no hard assignment

        GuidePdf pdfTopWalk;
        const int cluster = SampleSuperpixelClusterHeap(uint(selectedParent), walkXi.x, pdfTopWalk);
        const int clusterRootId = (cluster >= 0) ? gClusterRootNodes[cluster] : -1;
        if (clusterRootId < 0)
            guideOutcome = 3u;
        else
        {
            GuidePdf pdfTree;
            const int leafNodeId = TraverseLightTreeToLeaf(
                clusterRootId, litVoxelCount - 1u, walkXi.y,
                hit.position, surface.N, treeWeightMode, pdfTree);
            if (leafNodeId < 0)
            {
                // SIByL strategy 5 wipes the STRATEGY contributions on a dead
                // branch but adds direct light AFTER the wipe (gi.slang:512
                // wipe, :635 direct) — returning black here also discarded the
                // direct term, producing intermittent black samples (frame
                // flicker). Return the direct-only radiance instead, matching
                // SIByL's ordering. Rare with intensity-only traversal
                // (internal intensity = exact child sum), but not impossible
                // with float summation drift.
                return (debugView >= 3u) ? float3(0, 0, 0)
                                         : CalculateDirectLightning(hit, surface);
            }

            const uint compactID = uint(gLightTreeNodes[leafNodeId].voxelIndex);
            const int3 v = VoxelCoordFromFlatId(gVoxCompactIds[compactID]);

            float pdfDir;
            float3 aabbMin, aabbMax;
            float3 dir = SampleVoxelSolidAngle(
                hit.position, v, float3(faceXi, quadXi), pdfDir, aabbMin, aabbMax);
            // Cluster-selection probability = the fuzzy MIXTURE (not the single
            // walked parent's pdfTopWalk): the actual selection process is
            // "pick parent by weight, then walk its heap", whose density for a
            // cluster is sum_f w_f x leaf_f/root_f — the same formula the
            // reverse query uses, keeping the pair consistent (see
            // FuzzyClusterMixturePdf header).
            const GuidePdf pdfTop = FuzzyClusterMixturePdf(fuzzyWeights, fuzzyIndices, cluster);
            const GuidePdf pdfG = pdfTop * pdfTree * GuidePdf(pdfDir);

            if (pdfG <= 0.0)
                guideOutcome = 5u;
            else if (dot(dir, surface.N) <= 0.0)
                guideOutcome = 6u;
            else
            {
                float3 f = EvalBsdfBounce(surface, dir);
                if (all(f == 0))
                    guideOutcome = 7u;
                else
                {
                    float3 hitPos;
                    bool didHit;
                    float3 incoming = TraceIndirect(hit.position, dir, seed, /*writeVpl*/ false, hitPos, didHit);

                    // Semi-NEE gate: the claimed pdf belongs to the chosen
                    // voxel, so only count hits inside its AABB.
                    bool accepted = didHit && all(hitPos >= aabbMin) && all(hitPos <= aabbMax);
                    guideOutcome = accepted ? 2u : 1u;

                    if (accepted)
                    {
                        float pdfBAtDir = PdfBsdf(surface, specularProb, dir);
                        if (isnan(pdfBAtDir)) pdfBAtDir = 0.0; // SIByL w1 guard
                        float weight = MisWeight(float(pdfG), pdfBAtDir);
                        misWeightG = weight;
                        if (debugView < 3u)
                        {
                            float NdotL = dot(surface.N, dir);
                            radiance += f * NdotL * incoming * weight / float(pdfG);
                        }
                    }
                }
            }
        }
    }

    // MIS weight false-color: R = BSDF strategy weight, G = guide strategy
    // weight (at their respective sampled directions).
    if (debugView == 3u)
        radiance = float3(misWeightB, misWeightG, 0);

    // Guided sample acceptance/rejection reason (see guideOutcome codes above).
    if (debugView == 4u)
    {
        if (guideOutcome == 2u)      radiance = float3(0, 1, 0);    // accepted
        else if (guideOutcome == 1u) radiance = float3(1, 0, 0);    // gate reject
        else if (guideOutcome == 3u) radiance = float3(0, 1, 1);    // no cluster
        else if (guideOutcome == 5u) radiance = float3(1, 0.5, 0);  // pdf <= 0
        else if (guideOutcome == 6u) radiance = float3(0.5, 0, 1);  // below horizon
        else if (guideOutcome == 7u) radiance = float3(1, 1, 0);    // zero BRDF
        else                         radiance = float3(0, 0, 1);    // guide dead
    }

    return radiance;
}

// ---- Ray generation ----

// Shared integrator body; entered from the pipeline raygen or the compute
// (inline-RayQuery) wrapper below with gLaunchIndex/gLaunchDims already set.
void GuidedIntegratorMain()
{
    uint2 launchIndex = gLaunchIndex;
    uint2 dims = gLaunchDims;
    uint pixelId = launchIndex.x + launchIndex.y * dims.x;

#if GUIDING_DEBUG_VIEWS
    // View 7: per-pixel VPL buffer — decode its octahedral normal directly,
    // no rays needed. Black where the injection bounce missed (buffer zero).
    if (((guidingFlags >> 1) & 15u) == 7u)
    {
        float4 vpl = gVplPosition[launchIndex];
        float3 col = all(vpl == 0.0f)
            ? float3(0, 0, 0)
            : Unorm32OctahedronToUnitVector(asuint(vpl.w)) * 0.5f + 0.5f;
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }
#endif

    // ADR 0009: this raygen owns the per-pixel VPL slot — zero it up front so
    // pixels whose BSDF bounce misses (or never traces: sky, guide debug
    // views) stay empty for next frame's cvis, exactly like the dedicated
    // injection pass's per-pixel clear.
    if (voxReuseGiVpl != 0u)
        gVplPosition[launchIndex] = float4(0, 0, 0, 0);

    // First path vertex from the shared VBuffer (ADR 0004): reconstruct the
    // hit once; every spp sample shares it and diverges at the bounce.
    VBufferData vb = UnpackVBufferData(gVBuffer[launchIndex]);
    if (IsVBufferInvalid(vb))
    {
        // Sky pixel — sample the skybox along the same jittered primary
        // direction the VBuffer pass traced.
        float3 origin, direction;
        float2 jitter = VBufferPixelJitter(launchIndex, frameIndex, vbufferJitterEnabled);
        GenerateCameraRayJittered(launchIndex, jitter, float2(gLaunchDims), origin, direction);
        float u = atan2(direction.z, direction.x) / (2.0 * PI) + 0.5;
        float v = -asin(clamp(direction.y, -1.0, 1.0)) / PI + 0.5;
        float3 sky = g_skybox.SampleLevel(gsamLinearWrap, float2(u, v), 0).rgb;
        gOutput[launchIndex] = float4(sky, 1.0);
        return;
    }

    InstanceInfo instance = g_instanceInfo[vb.instanceId];
    GeometryInfo geometry = g_geometryInfo[instance.geometryIndex];
    HitData hit = GetHitData(vb.primitiveId, geometry.vertexOffset, geometry.indexOffset,
                             vb.barycentrics, instance.objectToWorld);

    float3 albedo = SampleTextureColor(instance, hit).rgb * instance.baseColorFactor.rgb;
    float2 rm = SampleRoughnessMetallic(instance, hit);
    float roughness = max(rm.x, MIN_ROUGHNESS);
    float metallic = rm.y;

    float3 N = SampleWorldSpaceNormal(instance, hit);
    float3 V = normalize(cameraWorldPos - hit.position);

    float3 geometricN = normalize(mul((float3x3)instance.objectToWorld, hit.tri_normal));
    if (dot(geometricN, V) < 0.0)
        N = -N;

    float NdotV = max(dot(N, V), 1e-4);  // div-by-zero guard only; 0.1 floored grazing specular

    SurfaceData surface;
    surface.N         = N;
    surface.V         = V;
    surface.NdotV     = NdotV;
    surface.F0        = lerp(DIELECTRIC_F0, albedo, metallic);
    surface.albedo    = albedo;
    surface.roughness = roughness;
    surface.metallic  = metallic;

#if GUIDING_DEBUG_VIEWS
    uint debugView = (guidingFlags >> 1) & 15u;
#else
    const uint debugView = 0u; // folds every view branch out of the hot path
#endif
    const uint treeWeightMode = (guidingFlags >> 5) & 3u;

#if GUIDING_DEBUG_VIEWS
    // View 5: inverse-index round-trip. For the primary hit's voxel, look up
    // gInverseIndex -> compactID and confirm gCompactIds maps back to the same
    // voxel. green = consistent, red = mismatch (bug), black = inactive voxel.
    if (debugView == 5u)
    {
        int3 v = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
        float3 col = float3(0, 0, 0);
        if (all(v >= 0) && all(v < int(voxGridDim)))
        {
            uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
            int ci = gVoxInverseIndex[flatId];
            if (ci >= 0)
                col = (gVoxCompactIds[ci] == flatId) ? float3(0, 1, 0) : float3(1, 0, 0);
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 6: representative VPL check. For the primary hit's ACTIVE voxel,
    // gVoxelRepresentative must hold a surface point inside that voxel.
    // Status colors only — normal display would collide with failure colors
    // once accumulation and tone mapping blend/skew them.
    // green = OK, red = active voxel with no data, magenta = stored position
    // outside the voxel (beyond FP-boundary tolerance), black = inactive.
    if (debugView == 6u)
    {
        int3 v = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
        float3 col = float3(0, 0, 0);
        if (all(v >= 0) && all(v < int(voxGridDim)))
        {
            uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
            if (gVoxInverseIndex[flatId] >= 0)
            {
                float4 representative = gVoxelRepresentative[v];
                // Writer classifies with floor((p-min)/size); this test
                // reconstructs the AABB by multiplication. The two round
                // differently, so allow a tiny boundary tolerance.
                const float boundaryTolerance = voxVoxelSize * 1e-4f;
                float3 voxelMin = voxGridMin + float3(v) * voxVoxelSize - boundaryTolerance;
                float3 voxelMax = voxelMin + voxVoxelSize + 2.0f * boundaryTolerance;
                if (all(representative == 0.0f))
                    col = float3(1, 0, 0);
                else if (any(representative.xyz < voxelMin) || any(representative.xyz > voxelMax))
                    col = float3(1, 0, 1);
                else
                    col = float3(0, 1, 0);
            }
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 8: voxel fingerprints. popcount(fingerprint)/128 as grayscale at the
    // primary hit's voxel (bright = sees many representatives). Green overlay =
    // the 128 stratified representative pixels this frame (recomputed with the
    // same hash the presample kernel used). magenta = inverse index points past
    // the compacted count (corruption). dark blue = unlit voxel / sky.
    if (debugView == 8u)
    {
        uint2 dims = gLaunchDims;
        uint randSeed = pcg_hash(frameIndex);
        float2 cellSize = float2(dims) / float2(16.0, 8.0);
        [loop] for (uint c = 0; c < 128u; ++c)
        {
            uint cx = c % 16u;
            uint cy = c / 16u;
            uint s = pcg_hash((c * 9781u + randSeed * 26699u) | 1u);
            int2 p = clamp(int2(cellSize * (float2(cx, cy) + Random2D(s))),
                           int2(0, 0), int2(dims) - int2(1, 1));
            if (all(uint2(p) == launchIndex))
            {
                gOutput[launchIndex] = float4(0, 1, 0, 1);
                return;
            }
        }

        float3 col = float3(0, 0, 0.15); // unlit / out of grid
        int3 v = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
        if (all(v >= 0) && all(v < int(voxGridDim)))
        {
            uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
            int ci = gVoxInverseIndex[flatId];
            if (ci >= 0)
            {
                if (uint(ci) >= gVoxCounters[0])
                {
                    col = float3(1, 0, 1); // compactID past the count = corruption
                }
                else
                {
                    uint bits = countbits(gVoxelFingerprints[ci * 4 + 0])
                              + countbits(gVoxelFingerprints[ci * 4 + 1])
                              + countbits(gVoxelFingerprints[ci * 4 + 2])
                              + countbits(gVoxelFingerprints[ci * 4 + 3]);
                    col = float3(float(bits) / 128.0, float(bits) / 128.0, float(bits) / 128.0);
                }
            }
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 9: cluster assignments. Categorical color per cluster id at the
    // primary hit's voxel — expect ~32 coherent patches whose colors reshuffle
    // every frame (compaction order churn, not a bug). white = one of the 32
    // seed voxels. magenta = bad inverse index or unassigned voxel. dark
    // blue = unlit voxel / sky.
    if (debugView == 9u)
    {
        float3 col = float3(0, 0, 0.15); // unlit / out of grid
        int3 v = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
        if (all(v >= 0) && all(v < int(voxGridDim)))
        {
            uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
            int ci = gVoxInverseIndex[flatId];
            if (ci >= 0)
            {
                int cluster = (uint(ci) < gVoxCounters[0]) ? gVoxelClusterAssignments[ci] : -1;
                bool isSeed = false;
                [loop] for (uint s = 0; s < 32u; ++s)
                {
                    if (gClusterSeedCompactIds[s] == ci) isSeed = true;
                }
                if (isSeed)
                {
                    col = float3(1, 1, 1);
                }
                else if (cluster < 0 || cluster >= 32)
                {
                    col = float3(1, 0, 1); // bad inverse index or unassigned
                }
                else
                {
                    // Saturated hue wheel; golden-ratio step spreads neighboring
                    // cluster ids far apart. Full-sat colors survive tone mapping
                    // (a hashed 0.25-floored palette washed out to white pastel).
                    float hue = frac(float(cluster) * 0.618034);
                    float h6 = hue * 6.0;
                    col = saturate(float3(abs(h6 - 3.0) - 1.0,
                                          2.0 - abs(h6 - 2.0),
                                          2.0 - abs(h6 - 4.0)));
                }
            }
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 10: cluster visibility. Grayscale = popcount(mask)/32 at this pixel's
    // superpixel tile — "how many of the 32 light clusters can this screen region
    // see". Bright in open areas, darker in occluded pockets. Blocky at 32px
    // tile granularity (expected). all-black = check kernel dead, all-white = OR
    // saturation bug.
    if (debugView == 10u)
    {
        uint2 maskCoord = launchIndex / 32u; // SUPERPIXEL_SIZE
        uint mask = gClusterVisibilityMask[maskCoord];
        float g = float(countbits(mask)) / 32.0;
        gOutput[launchIndex] = float4(g, g, g, 1.0);
        return;
    }

    // View 11: bottom light tree health. Walk gLightTreeNodes from the hit
    // voxel's leaf up to the root, checking the cluster's root node is on the
    // ancestor path. green = root reached + cluster root ancestor; yellow = root
    // reached but cluster root not ancestor; red = parent-walk never reached the
    // root (pointer cycle/corruption); magenta = bad compact->leaf / vx_idx
    // round-trip / overflow frame; dark blue = unlit/sky.
    if (debugView == 11u)
    {
        float3 col = float3(0, 0, 0.15); // unlit / out of grid
        int3 v = int3(floor((hit.position - voxGridMin) / voxVoxelSize));
        if (all(v >= 0) && all(v < int(voxGridDim)))
        {
            uint flatId = uint(v.x) + uint(v.y) * voxGridDim + uint(v.z) * voxGridDim * voxGridDim;
            int compactID = gVoxInverseIndex[flatId];
            if (compactID >= 0)
            {
                if (uint(compactID) >= gVoxCounters[0])
                {
                    col = float3(1, 0, 1); // lit voxel outside the compacted list
                }
                else
                {
                    int leaf = gCompactToLeaf[compactID];
                    bool badMapping = (leaf < 0) ||
                        (uint(gLightTreeNodes[leaf].voxelIndex) != uint(compactID)); // overflow / round-trip fail
                    if (badMapping)
                    {
                        col = float3(1, 0, 1);
                    }
                    else
                    {
                        int cluster = gVoxelClusterAssignments[compactID];
                        bool unassigned = (cluster < 0 || cluster >= 32);
                        int clusterRoot = unassigned ? -1 : gClusterRootNodes[cluster];
                        int cur = leaf;
                        bool onPath = false;
                        bool reachedRoot = false;
                        [loop] for (uint step = 0; step < 64u; ++step)
                        {
                            if (cur == clusterRoot) onPath = true;
                            uint parent = uint(gLightTreeNodes[cur].parentIndex);
                            if (parent == 0xFFFFu) { reachedRoot = true; break; }
                            cur = int(parent);
                        }
                        if (!reachedRoot)       col = float3(1, 0, 0);       // red: walk failed
                        else if (unassigned)    col = float3(0, 1, 1);       // cyan: voxel has no cluster (assignment -1)
                        else if (onPath)        col = float3(0, 1, 0);       // green: healthy
                        else                    col = float3(1, 1, 0);       // yellow: cluster root not ancestor
                    }
                }
            }
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 12: top-level tree heap health (screen-space, per superpixel). The
    // implicit binary heap must satisfy heap[i] == heap[2i]+heap[2i+1] for the 31
    // internal slots (slot 0 is dead). green = invariant holds + root > 0;
    // magenta = a parent != sum of its children (wave-reduction translation bug);
    // red = NaN/inf anywhere in the heap; dark blue = root 0 (superpixel sees no
    // lit cluster / sky). Blocky at 32px superpixel granularity (expected).
    if (debugView == 12u)
    {
        uint2 dims = gLaunchDims;
        uint mapX = (dims.x + 31u) / 32u; // SUPERPIXEL_SIZE
        uint2 spixel = launchIndex / 32u;
        uint base = (spixel.y * mapX + spixel.x) * 64u;

        bool anyBad = false;
        bool anyNaN = false;
        [loop] for (uint i = 1u; i < 32u; ++i)
        {
            float parent = gSpixelClusterImportanceHeap[base + i];
            float childSum = gSpixelClusterImportanceHeap[base + 2u * i]
                           + gSpixelClusterImportanceHeap[base + 2u * i + 1u];
            if (isnan(parent) || isinf(parent) || isnan(childSum) || isinf(childSum))
                anyNaN = true;
            if (abs(parent - childSum) > 1e-3 * max(abs(parent), 1e-6))
                anyBad = true;
        }
        [loop] for (uint l = 32u; l < 64u; ++l)
        {
            float v = gSpixelClusterImportanceHeap[base + l];
            if (isnan(v) || isinf(v)) anyNaN = true;
        }

        float root = gSpixelClusterImportanceHeap[base + 1u];
        float3 col;
        if (anyNaN)          col = float3(1, 0, 0);    // red: NaN/inf
        else if (anyBad)     col = float3(1, 0, 1);    // magenta: invariant broken
        else if (root > 0.0) col = float3(0, 1, 0);    // green: healthy
        else                 col = float3(0, 0, 0.15); // dark blue: empty superpixel
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 13: forward/reverse pdf round trip. Sample the discrete guide chain
    // once (heap walk + tree walk), then reverse-query the SAME outcome through
    // the lookup chain the BSDF-sample pdf uses. The two must telescope to the
    // same probability and map back to the same leaf/cluster — any mismatch is
    // a silent-bias bug the image alone would never show. green = match,
    // magenta = mismatch, red = NaN/inf, dark blue = guide dead here.
    if (debugView == 13u)
    {
        float3 col = float3(0, 0, 0.15); // guide dead / sky-adjacent
        const int spixelFlat = gSpixelIndexImage[launchIndex];
        const uint litVoxelCount = LitVoxelCount();
        if (litVoxelCount > 0u && spixelFlat >= 0)
        {
            uint seed = pcg_hash(pixelId ^ (frameIndex * 805459861u));
            float2 xi = Random2D(seed);
            GuidePdf pdfTop;
            const int cluster = SampleSuperpixelClusterHeap(uint(spixelFlat), xi.x, pdfTop);
            if (cluster >= 0)
            {
                const int clusterRootId = gClusterRootNodes[cluster];
                if (clusterRootId < 0)
                {
                    col = float3(1, 0, 1); // heap picked a cluster with no tree root
                }
                else
                {
                    GuidePdf pdfTree;
                    const int leafNodeId = TraverseLightTreeToLeaf(
                        clusterRootId, litVoxelCount - 1u, xi.y,
                        hit.position, N, treeWeightMode, pdfTree);
                    if (leafNodeId < 0)
                    {
                        col = float3(1, 0, 1); // dead branch under a live root
                    }
                    else
                    {
                        const uint compactID = uint(gLightTreeNodes[leafNodeId].voxelIndex);

                        // Reverse chain on the sampled outcome.
                        const uint heapBase = uint(spixelFlat) * 64u;
                        const float topTotal = gSpixelClusterImportanceHeap[heapBase + 1u];
                        const float topLeaf  = gSpixelClusterImportanceHeap[heapBase + 32u + uint(cluster)];
                        const GuidePdf pdfTopReverse = (topTotal == 0.0f) ? 0.0
                            : GuidePdf(topLeaf) / GuidePdf(topTotal);
                        const int leafBack    = gCompactToLeaf[compactID];
                        const int clusterBack = (compactID < litVoxelCount)
                            ? gVoxelClusterAssignments[compactID] : -1;
                        const GuidePdf pdfTreeReverse = (leafBack >= 0)
                            ? PdfTraverseLightTree(clusterRootId, leafBack, hit.position, N, treeWeightMode) : 0.0;

                        const float forward = float(pdfTop * pdfTree);
                        const float reverse = float(pdfTopReverse * pdfTreeReverse);
                        const bool anyNaN = isnan(forward) || isinf(forward)
                                         || isnan(reverse) || isinf(reverse);
                        const bool mismatch = (leafBack != leafNodeId)
                                           || (clusterBack != cluster)
                                           || abs(forward - reverse) > 1e-3f * max(forward, 1e-12f);
                        if (anyNaN)        col = float3(1, 0, 0);
                        else if (mismatch) col = float3(1, 0, 1);
                        else               col = float3(0, 1, 0);
                    }
                }
            }
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }

    // View 14: forward-selected cluster. One heap walk per pixel per frame,
    // painted in the view-9 hue wheel — expect superpixel-blocky regions whose
    // boundaries line up with view 10's tiles and whose colors strobe per frame
    // (frame-varying draw + recluster churn). magenta = heap picked a cluster
    // with no tree root; dark blue = guide dead here.
    if (debugView == 14u)
    {
        float3 col = float3(0, 0, 0.15);
        const int spixelFlat = gSpixelIndexImage[launchIndex];
        if (LitVoxelCount() > 0u && spixelFlat >= 0)
        {
            uint seed = pcg_hash(pixelId ^ (frameIndex * 805459861u));
            GuidePdf pdfTop;
            const int cluster = SampleSuperpixelClusterHeap(
                uint(spixelFlat), Random2D(seed).x, pdfTop);
            if (cluster >= 0)
            {
                if (gClusterRootNodes[cluster] < 0)
                {
                    col = float3(1, 0, 1);
                }
                else
                {
                    float hue = frac(float(cluster) * 0.618034);
                    float h6 = hue * 6.0;
                    col = saturate(float3(abs(h6 - 3.0) - 1.0,
                                          2.0 - abs(h6 - 2.0),
                                          2.0 - abs(h6 - 4.0)));
                }
            }
        }
        gOutput[launchIndex] = float4(col, 1.0);
        return;
    }
#endif // GUIDING_DEBUG_VIEWS

    float3 F = FresnelSchlick(NdotV, surface.F0);
    float specularProb = (F.r + F.g + F.b) / 3.0;

    // SLIC superpixel of this pixel (hard assignment, kept as the fuzzy
    // fall-through) and the fuzzy 4-nearest parent set. Weights are
    // renormalized after dropping parents whose importance heap is empty
    // (SIByL gi.slang:293-299); all-dropped => guide dead for this pixel.
    // Read once; every spp sample shares them.
    const int spixelFlat = gSpixelIndexImage[launchIndex];
    float4 fuzzyWeights = gFuzzyWeights[launchIndex];
    const int4 fuzzyIndices = gFuzzyIndices[launchIndex];
    [unroll] for (int f = 0; f < 4; ++f)
    {
        const bool parentAlive = fuzzyWeights[f] > 0.0 && fuzzyIndices[f] >= 0 &&
            gSpixelClusterImportanceHeap[uint(fuzzyIndices[f]) * 64u + 1u] > 0.0f;
        if (!parentAlive)
            fuzzyWeights[f] = 0.0;
    }
    const float fuzzyTotal = dot(fuzzyWeights, float4(1, 1, 1, 1));
    if (fuzzyTotal > 0.0)
        fuzzyWeights /= fuzzyTotal;

    float3 accumulated = float3(0, 0, 0);
    for (uint i = 0; i < (uint)samplesPerPixel; i++)
    {
        uint seed = pcg_hash(pixelId ^ (i * 2654435761u) ^ (frameIndex * 805459861u));
        seed = pcg_hash(seed);
        accumulated += ShadeFirstVertex(hit, surface, specularProb, debugView,
                                        fuzzyWeights, fuzzyIndices, spixelFlat, seed);
    }

    gOutput[launchIndex] = float4(accumulated / samplesPerPixel, 1.0);
}

// ---- Entry points ----

#ifdef GUIDED_TRACE_RQ

// Compute wrapper (inline-RayQuery integrator, ADR 0011). One thread per
// pixel, same body as the pipeline raygen.
[numthreads(8, 8, 1)]
// Wave size left to the driver: measured 840 (default) vs 827 (wave32) vs 799 (wave64) frames/3s on RDNA, Deep Light b1.
void GuidedRqMain(uint3 dtid : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    if (dtid.x >= width || dtid.y >= height)
        return;
    gLaunchIndex = dtid.xy;
    gLaunchDims = uint2(width, height);
    GuidedIntegratorMain();
}

#else

[shader("raygeneration")]
void GuidedRayGen()
{
    gLaunchIndex = DispatchRaysIndex().xy;
    gLaunchDims = DispatchRaysDimensions().xy;
    GuidedIntegratorMain();
}

#endif // GUIDED_TRACE_RQ

#ifndef GUIDED_TRACE_RQ // pipeline-only hit/miss shaders

// ---- Miss ----

[shader("miss")]
void GuidedMiss(inout GuidedPayload payload : SV_RayPayload)
{
    // Sky shading happens in the raygen bounce loop (IndirectSkyRadiance).
    payload.hitFlag = 0;
}

// ---- Any hit (alpha cutout) ----

[shader("anyhit")]
void GuidedAnyHit(inout GuidedPayload payload : SV_RayPayload, in Attributes attr)
{
    InstanceInfo instance = g_instanceInfo[InstanceID()];
    uint vertexOffset = g_geometryInfo[instance.geometryIndex].vertexOffset;
    uint indexOffset = g_geometryInfo[instance.geometryIndex].indexOffset;
    HitData hit = GetHitData(PrimitiveIndex(), vertexOffset, indexOffset, attr.barycentrics);

    float4 albedo = SampleTextureColor(hit) * instance.baseColorFactor;
    if (albedo.a < EPSILON)
    {
        IgnoreHit();
    }
}

// ---- Closest hit ----

[shader("closesthit")]
void GuidedHit(inout GuidedPayload payload : SV_RayPayload, in Attributes attr)
{
    // Report-only (ADR 0007): shading happens in the raygen bounce loop.
    payload.instanceId = InstanceID();
    payload.primitiveId = PrimitiveIndex();
    payload.barycentrics = attr.barycentrics;
    payload.hitFlag = 1;
}

#endif // GUIDED_TRACE_RQ (pipeline-only hit/miss shaders)

#endif // GUIDED_PATH_TRACING_HLSL
