#pragma once

// Debug visualization for the guided path tracing technique.
// Encoded into PassConstants::guidingFlags bits 1-3 (see guidedPathTracing.hlsl).
enum class GuidingDebugView : int
{
	None = 0,
	BsdfStrategyOnly = 1,
	GuideStrategyOnly = 2,
	MisWeights = 3,       // R = BSDF strategy weight, G = guide strategy weight
	GuideAcceptance = 4,  // green = guided sample accepted, red = gate-rejected, blue = horizon/pdf-rejected
};
