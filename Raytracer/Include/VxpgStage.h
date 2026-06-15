#pragma once

// How far down the linear VXPG pipeline a frame must run. Ordered: a higher
// stage implies every lower stage also runs (voxelize -> inject -> build ->
// cluster). Both debug views and raytracing techniques declare the furthest
// stage they need; the renderer runs the pipeline up to the maximum.
enum class VxpgStage : int
{
	None = 0,
	Voxelize,
	Inject,
	GuidingBuild,
	Supervoxel,
	Superpixel,
};

inline bool operator>=(VxpgStage a, VxpgStage b)
{
	return static_cast<int>(a) >= static_cast<int>(b);
}
