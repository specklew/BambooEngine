// Per-tile adaptive guide-selection probability update (ADR 0015). Runs once
// after every guided one-sample dispatch: share = guide-attributed luminance /
// total strategy luminance this frame (both sums estimate their integrals
// through the contribution's 1/(q x pdf) scaling, so the ratio targets
// int(w_G f) / int(f) — the variance-aware q for the one-sample model).
// EMA-damped, clamped so both strategies stay explored; stats cleared for the
// next frame. Bound through the guided PT global root signature (u22/u23).

RWStructuredBuffer<float> gTileGuideQ        : register(u22);
RWStructuredBuffer<uint>  gTileStrategyStats : register(u23);

// Keep in sync with guidedPathTracing.hlsl's one-sample block.
#define ADAPTIVE_Q_MIN 0.05
#define ADAPTIVE_Q_MAX 0.5
#define ADAPTIVE_Q_EMA 0.25

[numthreads(64, 1, 1)]
void UpdateTileGuideQ(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint tileCount, stride;
    gTileGuideQ.GetDimensions(tileCount, stride);
    const uint tileId = dispatchThreadId.x;
    if (tileId >= tileCount)
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
