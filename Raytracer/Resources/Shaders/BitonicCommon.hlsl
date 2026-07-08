#ifndef BITONIC_COMMON_HLSL
#define BITONIC_COMMON_HLSL

// MiniEngine-style bitonic sort primitives, ported from SIByL
// bitonicsort/bitonic_common.hlsli. Fixed to uint64_t keys (the VXPG light-tree
// sort key = (clusterID<<48)|(30-bit Morton<<16)|leafID) — REQUIRES device
// support for Int64ShaderOps.

// ~uint64_t(0), NOT a hex literal: DXC silently truncates the "uL" suffix to
// 32 bits, which made the null key 0x00000000FFFFFFFF — SMALLER than real sort
// keys, so it sorted into the valid range and corrupted the tree.
static const uint64_t BITONIC_NULL_KEY = ~uint64_t(0);

// Widens Value by one bit at the location of the (single) set bit in OneBitMask,
// inserting a 1 there. SIByL InsertOneBit.
uint InsertOneBit(uint value, uint oneBitMask)
{
    uint mask = oneBitMask - 1;
    return (value & ~mask) << 1 | (value & mask) | oneBitMask;
}

// Ascending order. SIByL ShouldSwap.
bool ShouldSwap(uint64_t a, uint64_t b)
{
    return a > b;
}

#endif // BITONIC_COMMON_HLSL
