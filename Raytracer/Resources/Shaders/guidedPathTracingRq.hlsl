// Compute-shader (inline RayQuery) build of the guided integrator (ADR 0011).
// Same body as guidedPathTracing.hlsl; the define swaps the trace backend
// (TraceBounceRay / TraceShadow) and the entry point (GuidedRqMain).
#define GUIDED_TRACE_RQ 1
#include "guidedPathTracing.hlsl"
