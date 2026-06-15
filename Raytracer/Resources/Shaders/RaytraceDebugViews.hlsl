#ifndef RAYTRACE_DEBUG_VIEWS_HLSL
#define RAYTRACE_DEBUG_VIEWS_HLSL

// Include after RaytraceDebugMode.h. Mode integers mirror the RaytraceDebugMode enum.
bool TryRaytraceDebugView(int mode, RtDebugData d, out float3 color)
{
	color = float3(0, 0, 0);

	if (mode == 1) // HitHealth
	{
		if (any(isnan(d.N)))             color = float3(1, 0, 1); // NaN normal
		else if (any(isnan(d.position))) color = float3(0, 1, 1); // NaN position
		else                             color = float3(0, 1, 0); // healthy
		return true;
	}

	return false;
}

// Classifies a NaN bounce direction by cause. roughness via max(NaN, MIN_ROUGHNESS) stays NaN.
// The basis-collapse case: BuildONB's normalize(cross(up, N)) is NaN when N is non-unit or axis-aligned.
float3 BounceHealthColor(float3 bounceDir, float3 N, float roughness)
{
	if (!any(isnan(bounceDir))) return float3(0, 1, 0); // healthy
	if (any(isnan(N)))          return float3(1, 0, 0); // NaN normal
	if (isnan(roughness))       return float3(1, 1, 0); // NaN roughness
	if (abs(dot(N, N) - 1.0) > 1e-2) return float3(0, 1, 1); // N not unit

	float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
	if (dot(cross(up, N), cross(up, N)) < 1e-10) return float3(0, 0, 1); // basis collapse
	return float3(1, 0, 1); // sampling math
}

#endif // RAYTRACE_DEBUG_VIEWS_HLSL
