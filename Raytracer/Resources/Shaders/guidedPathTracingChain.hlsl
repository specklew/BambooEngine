// Compute build of the forward guide chain (ADR 0023). Same body as guidedPathTracing.hlsl; the
// define brings in the GuideChainMain entry point that writes the hand-off the raygen reads under
// GUIDE_CHAIN_IN_PASS. Shares the integrator's root signature, like the inline-RayQuery build does.
#define GUIDE_CHAIN_ENTRY 1
#include "guidedPathTracing.hlsl"
