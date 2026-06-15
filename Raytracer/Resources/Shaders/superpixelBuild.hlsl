// VXPG V2 Stage B: superpixel clustering (SLIC over the ShadingPoints G-buffer).
// Ported from SIByLEngine2023 addon/vxguiding/spixel/ (find-center + sum-center),
// with the disabled averaging center-update RE-ENABLED (see docs/adr/0002). Three
// entry points dispatched per frame:
//   InitSeedCenters -> ITER x [ FindCenterAssociation(gather=0) -> SumCenter ]
//                   -> ClearCounter -> FindCenterAssociation(gather=1)
// "color" term = primary-hit world position (ShadingPoints.xyz); normal gate uses
// the octahedral-packed normal in .w. Tile-local averaging (approximation, ADR-0002).

#include "Octahedral.hlsl"

cbuffer SuperpixelCB : register(b0)
{
    int2  map_size;        // ceil(img_size / spixel_size)
    int2  img_size;        // screen resolution
    int   spixel_size;     // SUPERPIXEL_SIZE (32)
    float weight;          // coherence weight (CVar superpixel.weight)
    float max_xy_dist;     // squared screen-xy normalizer
    float max_color_dist;  // squared position normalizer (CVar superpixel.posNormalizer)
    uint  writeGather;     // 1 = also emit gather lists + counter (final association only)
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
}

// ShadingPoints read as UAV to avoid a per-frame UAV<->SRV transition.
RWTexture2D<float4> u_input         : register(u0); // primary worldPos.xyz + octaN.w
RWTexture2D<float4> u_center        : register(u1); // representative pos + octaN (map_size)
RWTexture2D<int>    u_index         : register(u2); // per-pixel superpixel id
RWTexture2D<int4>   u_fuzzyIdx       : register(u3); // 4-nearest superpixel ids
RWTexture2D<float4> u_fuzzyWeight    : register(u4); // 4-nearest weights
RWTexture2D<uint>   u_spixel_counter : register(u5); // pixels per superpixel (map_size)
RWTexture2D<int2>   u_spixel_gathered: register(u6); // gathered pixel coords (map*spixel_size)

static const float SP_INVALID_POS = 1e29; // ShadingPoints sentinel is 1e30

bool IsValidPixel(float4 sp) { return sp.x < SP_INVALID_POS; }

// ---- FuzzyVec: 4 nearest superpixels (free functions; HLSL has no mutating methods) ----

struct FuzzyVec
{
    int4   center; // 4 nearest superpixel ids (dynamic-indexed)
    float4 dist2;
    int    size;
};

FuzzyVec FuzzyInit()
{
    FuzzyVec v;
    v.center = int4(-1, -1, -1, -1);
    v.dist2  = float4(0, 0, 0, 0);
    v.size   = 0;
    return v;
}

int FuzzyMaxSlot(FuzzyVec v)
{
    int maxIdx = 0;
    float maxDist = v.dist2[0];
    [unroll] for (int i = 1; i < 4; ++i)
        if (v.dist2[i] > maxDist) { maxDist = v.dist2[i]; maxIdx = i; }
    return maxIdx;
}

void FuzzyInsert(inout FuzzyVec v, int c, float d2)
{
    if (v.size < 4)
    {
        v.center[v.size] = c;
        v.dist2[v.size]  = d2;
        v.size++;
    }
    else
    {
        int maxIdx = FuzzyMaxSlot(v);
        if (d2 < v.dist2[maxIdx]) { v.center[maxIdx] = c; v.dist2[maxIdx] = d2; }
    }
}

void FuzzyPack(FuzzyVec v, int2 idxImg)
{
    float4 w = float4(0, 0, 0, 0);
    int4   c = int4(-1, -1, -1, -1);
    if (v.size == 0)
    {
        u_fuzzyIdx[idxImg]    = c;
        u_fuzzyWeight[idxImg] = w;
        return;
    }
    int zeroIdx = -1;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        if (i < v.size)
        {
            if (v.dist2[i] == 0) zeroIdx = i;
            w[i] = 1.0 / v.dist2[i];
            c[i] = v.center[i];
        }
    }
    if (zeroIdx != -1) { w = float4(0, 0, 0, 0); w[zeroIdx] = 1; }
    w /= dot(w, float4(1, 1, 1, 1));
    u_fuzzyIdx[idxImg]    = c;
    u_fuzzyWeight[idxImg] = w;
}

// SLIC distance: .x = normal-gated (1e6 if facing away), .y = ungated fallback.
float2 ComputeSlicDistance(
    float3 pixPos, float3 pixNormal, float2 pixXy,
    float3 spPos,  float3 spNormal,  float2 spXy,
    float w, float normXy, float normColor)
{
    float  dotN  = dot(pixNormal, spNormal);
    float3 dCol  = pixPos - spPos;
    float  dcolor = dot(dCol, dCol);
    float2 dXy   = pixXy - spXy;
    float  dxy   = dot(dXy, dXy);
    float  retval = dcolor * normColor + w * dxy * normXy;
    return float2((dotN > 0.01) ? retval : 1000000.0, retval);
}

// Tile pixel-space center, nudged inward when the tile straddles the image edge.
int2 SpixelImageCenter(int2 sp)
{
    int img_x = sp.x * spixel_size + spixel_size / 2;
    int img_y = sp.y * spixel_size + spixel_size / 2;
    img_x = img_x >= img_size.x ? (sp.x * spixel_size + img_size.x) / 2 : img_x;
    img_y = img_y >= img_size.y ? (sp.y * spixel_size + img_size.y) / 2 : img_y;
    return int2(img_x, img_y);
}

// ---- Pass 1: seed centers from the tile middle pixel ----

[numthreads(8, 8, 1)]
void InitSeedCenters(uint3 tid : SV_DispatchThreadID)
{
    int2 sp = int2(tid.xy);
    if (any(sp >= map_size)) return;
    u_center[sp] = u_input[SpixelImageCenter(sp)];
}

// ---- Pass 2: per-pixel association + fuzzy 4-nearest + (final) gather ----

groupshared float3 gs_pos[9];
groupshared float3 gs_nrm[9];
groupshared float2 gs_xy[9];
groupshared int    gs_id[9];
groupshared uint   gs_counter[9];
groupshared uint   gs_offset[9];

int Offset2Idx(int2 o) { return (o.y + 1) * 3 + o.x + 1; }

[numthreads(16, 16, 1)]
void FindCenterAssociation(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex)
{
    int2 idxImg = int2(dtid.xy);
    // All 16x16 threads of a group share one ctr (16-wide span stays inside one 32-tile).
    int ctr_x = idxImg.x / spixel_size;
    int ctr_y = idxImg.y / spixel_size;

    if (gid < 9)
    {
        int2 offset = int2((int)gid % 3 - 1, (int)gid / 3 - 1);
        int idx = Offset2Idx(offset);
        int cx = ctr_x + offset.x;
        int cy = ctr_y + offset.y;
        if (cx >= 0 && cy >= 0 && cx < map_size.x && cy < map_size.y)
        {
            float4 ci = u_center[int2(cx, cy)];
            gs_pos[idx] = ci.xyz;
            gs_nrm[idx] = Unorm32OctahedronToUnitVector(asuint(ci.w));
            gs_xy[idx]  = float2(SpixelImageCenter(int2(cx, cy)));
            gs_id[idx]  = cy * map_size.x + cx;
        }
        else
        {
            gs_id[idx] = -1;
        }
        gs_counter[gid] = 0;
        gs_offset[gid]  = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    bool oob = any(idxImg >= img_size);
    float4 pix = oob ? float4(SP_INVALID_POS + 1, 0, 0, 0) : u_input[idxImg];
    float3 pixN = Unorm32OctahedronToUnitVector(asuint(pix.w));
    bool valid = IsValidPixel(pix) && !oob;

    int   minidx = -1;     float dist = 999999.0;
    int   minidxF = -1;    float distF = 999999.0;
    FuzzyVec fv  = FuzzyInit();
    FuzzyVec fvF = FuzzyInit();

    [unroll] for (int n = 0; n < 9; ++n)
    {
        int j = n % 3 - 1;
        int i = n / 3 - 1;
        int cx = ctr_x + j;
        int cy = ctr_y + i;
        int off = Offset2Idx(int2(j, i));
        int spId = gs_id[off];
        if (valid && cx >= 0 && cy >= 0 && cx < map_size.x && cy < map_size.y && spId >= 0)
        {
            float2 cd = ComputeSlicDistance(
                pix.xyz, pixN, float2(idxImg),
                gs_pos[off], gs_nrm[off], gs_xy[off],
                weight, max_xy_dist, max_color_dist);
            if (cd.x < dist)  { dist = cd.x;  minidx  = spId; }
            if (cd.y < distF) { distF = cd.y; minidxF = spId; }
            FuzzyInsert(fv,  spId, cd.x);
            FuzzyInsert(fvF, spId, cd.y);
        }
    }

    bool gateValid = (minidx >= 0);
    int spixelID = gateValid ? minidx : minidxF;
    u_index[idxImg] = valid ? spixelID : -1;
    if (minidxF == -1) fvF.size = 0;
    FuzzyVec chosen = fv;
    if (!gateValid) chosen = fvF;
    FuzzyPack(chosen, idxImg);

    if (writeGather == 0) return;

    // Gather: bin this pixel into its superpixel's pixel-list (cap spixel_size^2).
    int2 spId2D = int2(-1, -1);
    int  gsSlot = 0;
    bool contributes = valid && spixelID >= 0;
    if (contributes)
    {
        spId2D = int2(spixelID % map_size.x, spixelID / map_size.x);
        int2 offSp = (spId2D - int2(ctr_x, ctr_y)) + int2(1, 1);
        gsSlot = offSp.x + offSp.y * 3;
    }

    uint localOffset = 0;
    if (contributes && gsSlot >= 0 && gsSlot < 9)
        InterlockedAdd(gs_counter[gsSlot], 1, localOffset);

    GroupMemoryBarrierWithGroupSync();

    if (gid < 9)
    {
        int2 offSp = int2((int)gid % 3, (int)gid / 3) - int2(1, 1);
        int2 sp2D  = int2(ctr_x, ctr_y) + offSp;
        if (all(sp2D >= 0) && all(sp2D < map_size) && gs_counter[gid] > 0)
            InterlockedAdd(u_spixel_counter[sp2D], gs_counter[gid], gs_offset[gid]);
    }

    GroupMemoryBarrierWithGroupSync();

    if (contributes && gsSlot >= 0 && gsSlot < 9)
    {
        uint flat = gs_offset[gsSlot] + localOffset;
        if (flat < uint(spixel_size * spixel_size))
        {
            int2 taskOffset = spId2D * spixel_size;
            int2 sub = int2(int(flat) % spixel_size, int(flat) / spixel_size);
            u_spixel_gathered[taskOffset + sub] = idxImg;
        }
    }
}

// ---- Pass 3: tile-local averaged center update (groupshared reduction, ADR-0002) ----
// One 32x32 group per superpixel tile reduces its valid pixels to an averaged
// representative. Approximation: averages pixels OF the tile, not pixels assigned.

groupshared float3 red_pos[32 * 32];
groupshared float3 red_nrm[32 * 32];
groupshared uint   red_cnt[32 * 32];

[numthreads(32, 32, 1)]
void SumCenter(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    int2 pixel = int2(dtid.xy);
    int2 sp = pixel / spixel_size;

    bool oob = any(pixel >= img_size);
    float4 ci = oob ? float4(SP_INVALID_POS + 1, 0, 0, 0) : u_input[pixel];
    bool valid = IsValidPixel(ci) && !oob;

    red_pos[gi] = valid ? ci.xyz : float3(0, 0, 0);
    red_nrm[gi] = valid ? Unorm32OctahedronToUnitVector(asuint(ci.w)) : float3(0, 0, 0);
    red_cnt[gi] = valid ? 1u : 0u;

    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = (32 * 32) / 2; stride > 0; stride >>= 1)
    {
        if (gi < stride)
        {
            red_pos[gi] += red_pos[gi + stride];
            red_nrm[gi] += red_nrm[gi + stride];
            red_cnt[gi] += red_cnt[gi + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0 && red_cnt[0] > 0u)
    {
        float3 avgPos = red_pos[0] / float(red_cnt[0]);
        float3 avgNrm = normalize(red_nrm[0] / float(red_cnt[0]));
        u_center[sp] = float4(avgPos, asfloat(UnitVectorToUnorm32Octahedron(avgNrm)));
    }
}

// ---- Pass 4: clear the per-superpixel counter before the final association ----

[numthreads(8, 8, 1)]
void ClearCounter(uint3 tid : SV_DispatchThreadID)
{
    int2 sp = int2(tid.xy);
    if (any(sp >= map_size)) return;
    u_spixel_counter[sp] = 0u;
}
