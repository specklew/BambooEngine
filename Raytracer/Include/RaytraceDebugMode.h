// RT debug visualizations, separate from RasterDebugMode. Both never run at once, so
// PassConstants.debugMode (one int) carries whichever mode is active.
#ifdef __cplusplus
#pragma once

#include "DebugViewDoc.h"

enum class RaytraceDebugMode : int
{
	None = 0,
	HitHealth = 1,
	BounceHealth = 2,
};

// Runtime docs, one per enum entry in order (FormatDebugViewDocs static_asserts the count).
inline constexpr DebugViewDoc kRaytraceDebugModeDocs[] = {
	{"nothing", "normal path-traced image", "debug path disabled"},
	{"PathTracingPass hit shading (NaN sentinel)", "all green; magenta = NaN normal, cyan = NaN position", "classifies NaNs in the closest-hit surface setup"},
	{"PathTracingPass bounce sampling (NaN sentinel)", "all green; other colors = NaN bounce direction by cause", "classifies NaNs in the sampled bounce direction (see BounceHealthColor)"},
};

#else // HLSL

struct RtDebugData
{
	float3 N;
	float3 position;
};

#endif
