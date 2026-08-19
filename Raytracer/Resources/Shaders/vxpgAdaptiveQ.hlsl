// Per-tile adaptive guide-selection probability update (ADR 0015). Runs once
// after every guided one-sample dispatch: share = guide-attributed luminance /
// total strategy luminance this frame (both sums estimate their integrals
// through the contribution's 1/(q x pdf) scaling, so the ratio targets
// int(w_G f) / int(f) — the variance-aware q for the one-sample model).
// EMA-damped, clamped so both strategies stay explored; stats cleared for the
// next frame. Bound through the guided PT global root signature (u22/u23).
// Measured 2026-08-19: on veach-ajar the guide carries well over half the
// energy everywhere, so the target sits on ADAPTIVE_Q_MAX and q is a flat 0.5 —
// adaptivity only bites where the guide's share drops below the cap.

#include "PassRegisters.h"

RWStructuredBuffer<float> gTileGuideQ        : BAMBOO_PASS_UAV(GUIDED_REG_TILE_GUIDE_Q);
RWStructuredBuffer<uint>  gTileStrategyStats : BAMBOO_PASS_UAV(GUIDED_REG_TILE_STRATEGY_STATS);

// Both buffers are bound as root descriptors, which carry no size — GetDimensions
// on them returns garbage and would let the tail of the last thread group run off
// the end. The CPU knows the tile grid, so it says so.
cbuffer AdaptiveQCB : BAMBOO_PASS_CBV(GUIDED_REG_ADAPTIVE_Q_CB)
{
    uint gTileCount;
};

// Keep in sync with guidedPathTracing.hlsl's one-sample block.
#define ADAPTIVE_Q_MIN 0.05
#define ADAPTIVE_Q_MAX 0.5
#define ADAPTIVE_Q_EMA 0.25

[numthreads(64, 1, 1)]
void UpdateTileGuideQ(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tileId = dispatchThreadId.x;
    if (tileId >= gTileCount)
        return;

    const uint guideLumFx = gTileStrategyStats[tileId * 2u + 0u];
    const uint totalLumFx = gTileStrategyStats[tileId * 2u + 1u];
    gTileStrategyStats[tileId * 2u + 0u] = 0u;
    gTileStrategyStats[tileId * 2u + 1u] = 0u;

    if (totalLumFx == 0u)
        return; // no guide-alive strategy energy this frame — hold q

    const float share = float(guideLumFx) / float(totalLumFx);
    float previous = gTileGuideQ[tileId];
    if (!(previous >= ADAPTIVE_Q_MIN && previous <= ADAPTIVE_Q_MAX))
        previous = 0.5; // fresh or garbage buffer — one update self-heals
    const float target = clamp(share, ADAPTIVE_Q_MIN, ADAPTIVE_Q_MAX);
    gTileGuideQ[tileId] = lerp(previous, target, ADAPTIVE_Q_EMA);
}
