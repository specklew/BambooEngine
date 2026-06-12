#pragma once

// Debug visualization for the guided path tracing technique.
// Encoded into PassConstants::guidingFlags bits 1-2 (see guidedPathTracing.hlsl).
enum class GuidingDebugView : int
{
	None = 0,
	BsdfStrategyOnly = 1,
	GuideStrategyOnly = 2,
	MisWeights = 3, // R = BSDF strategy weight, G = guide strategy weight
};
