// GPU bitonic sort over uint64 keys, ported from SIByL bitonicsort/
// {presort,outersort,innersort}-pass.slang (INDIRECT_DISPATCH variant). The
// element count is read live from a counter buffer each dispatch; Bamboo drives
// the 65536-element network with fixed worst-case dispatches (32 groups each)
// and lets the in-shader ListCount guards early-out (ADR 0003 option b).
//
// Used by the VXPG light tree to sort leaf codes so each cluster becomes a
// contiguous Morton-ordered run. Ported from SIByL; identifiers renamed.

#include "BitonicCommon.hlsl"

RWStructuredBuffer<uint64_t> gSortBuffer : register(u0); // SIByL g_SortBuffer

// The light-tree dispatch-args buffer, addressed raw; the clamped valid leaf
// count lives at byte offset gCounterOffset (SIByL u_CounterBuffer).
RWByteAddressBuffer gCounter : register(u1);

cbuffer BitonicCB : register(b0)
{
    uint gK; // outer/inner stage size (>= 4096); unused by presort
    uint gJ; // outer sub-stage (>= 2048 && < k); unused by presort/inner
    uint gCounterOffset; // byte offset of the valid-count field in gCounter
}

uint LoadCount()
{
    return gCounter.Load(gCounterOffset);
}

// ---- PresortKeys (presort-pass) -------------------------------------------
// Each 1024-thread group sorts one 2048-element block into bitonic order.

groupshared uint64_t sPresortValues[2048];

void PresortLoad(uint index, uint listCount)
{
    sPresortValues[index & 2047] = (index < listCount) ? gSortBuffer[index] : BITONIC_NULL_KEY;
}

void PresortSave(uint index, uint listCount)
{
    if (index < listCount)
        gSortBuffer[index] = sPresortValues[index & 2047];
}

[numthreads(1024, 1, 1)]
void PresortKeys(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    const uint offset = gid.x * 2048;
    const uint listCount = LoadCount();

    PresortLoad(offset + tid, listCount);
    PresortLoad(offset + tid + 1024, listCount);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint k = 2; k <= 2048; k <<= 1)
    {
        [unroll]
        for (uint j = k / 2; j > 0; j /= 2)
        {
            uint index2 = InsertOneBit(tid, j);
            uint index1 = index2 ^ (k == 2 * j ? k - 1 : j);

            uint64_t a = sPresortValues[index1];
            uint64_t b = sPresortValues[index2];
            if (ShouldSwap(a, b))
            {
                sPresortValues[index1] = b;
                sPresortValues[index2] = a;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    PresortSave(offset + tid, listCount);
    PresortSave(offset + tid + 1024, listCount);
}

// ---- OuterSortStep (outersort-pass) ---------------------------------------
// One compare-exchange per thread across the whole buffer for stage (k, j).

[numthreads(1024, 1, 1)]
void OuterSortStep(uint3 tid : SV_DispatchThreadID)
{
    const uint numElements = LoadCount();

    uint index2 = InsertOneBit(tid.x, gJ);
    uint index1 = index2 ^ (gK == 2 * gJ ? gK - 1 : gJ);
    if (index2 >= numElements)
        return;

    uint64_t a = gSortBuffer[index1];
    uint64_t b = gSortBuffer[index2];
    if (ShouldSwap(a, b))
    {
        gSortBuffer[index1] = b;
        gSortBuffer[index2] = a;
    }
}

// ---- InnerSortStep (innersort-pass) ---------------------------------------
// Finishes stage k inside each 2048-element block via groupshared memory.

groupshared uint64_t sInnerValues[2048];

void InnerLoad(uint element, uint listCount)
{
    sInnerValues[element & 2047] = (element < listCount) ? gSortBuffer[element] : BITONIC_NULL_KEY;
}

void InnerStore(uint element, uint listCount)
{
    if (element < listCount)
        gSortBuffer[element] = sInnerValues[element & 2047];
}

[numthreads(1024, 1, 1)]
void InnerSortStep(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    const uint groupStart = gid.x * 2048;
    const uint numElements = LoadCount();

    InnerLoad(groupStart + gi, numElements);
    InnerLoad(groupStart + gi + 1024, numElements);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint j = 1024; j > 0; j /= 2)
    {
        uint index2 = InsertOneBit(gi, j);
        uint index1 = index2 ^ j;

        uint64_t a = sInnerValues[index1];
        uint64_t b = sInnerValues[index2];
        if (ShouldSwap(a, b))
        {
            sInnerValues[index1] = b;
            sInnerValues[index2] = a;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    InnerStore(groupStart + gi, numElements);
    InnerStore(groupStart + gi + 1024, numElements);
}
