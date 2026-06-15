// RT debug visualizations, separate from RasterDebugMode. Both never run at once, so
// PassConstants.debugMode (one int) carries whichever mode is active.
#ifdef __cplusplus
#pragma once

enum class RaytraceDebugMode : int
{
	None = 0,
	HitHealth = 1,    // magenta = NaN normal, cyan = NaN position, green = healthy
	BounceHealth = 2, // NaN bounce direction classified by cause (see BounceHealthColor)
};

#else // HLSL

struct RtDebugData
{
	float3 N;
	float3 position;
};

#endif
