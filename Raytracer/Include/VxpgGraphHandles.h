#pragma once
#include "RenderGraph.h"

// This frame's imports of every VXPG product, refreshed by Renderer::BuildVxpgGraph.
// Imports last exactly one frame (ADR 0017 §A2), so the whole struct is rebuilt
// every Render() and every handle is Invalid on a frame with no guiding subgraph.
// A technique reads it through FrameGraphContext to declare what it consumes —
// which is what keeps the producers behind it alive through culling.
struct VxpgGraphHandles
{
    GraphResourceHandle vbuffer              = InvalidGraphResource;
    GraphResourceHandle shadingPoints        = InvalidGraphResource;
    GraphResourceHandle voxelRepresentative  = InvalidGraphResource;
    GraphResourceHandle vplPosition          = InvalidGraphResource;
    GraphResourceHandle voxelOccupancy       = InvalidGraphResource;
    GraphResourceHandle bakedBoundMin        = InvalidGraphResource;
    GraphResourceHandle bakedBoundMax        = InvalidGraphResource;
    GraphResourceHandle voxelIrradiance      = InvalidGraphResource;
    GraphResourceHandle voxelVplCount        = InvalidGraphResource;
    GraphResourceHandle counters             = InvalidGraphResource;
    GraphResourceHandle compactIds           = InvalidGraphResource;
    GraphResourceHandle inverseIndex         = InvalidGraphResource;
    GraphResourceHandle compactLightPoints   = InvalidGraphResource;
    GraphResourceHandle premulIrradiance     = InvalidGraphResource;
    GraphResourceHandle liveBoundMin         = InvalidGraphResource;
    GraphResourceHandle liveBoundMax         = InvalidGraphResource;
    GraphResourceHandle voxelFingerprints    = InvalidGraphResource;
    GraphResourceHandle screenRepresentatives = InvalidGraphResource;
    GraphResourceHandle guidingDispatchArgs   = InvalidGraphResource;
    GraphResourceHandle clusterAssignments   = InvalidGraphResource;
    GraphResourceHandle clusterSeedCompactIds = InvalidGraphResource;
    GraphResourceHandle clusterCenters        = InvalidGraphResource;
    GraphResourceHandle superpixelIndex      = InvalidGraphResource;
    GraphResourceHandle superpixelCenter     = InvalidGraphResource;
    GraphResourceHandle superpixelCounter    = InvalidGraphResource;
    GraphResourceHandle superpixelGathered   = InvalidGraphResource;
    GraphResourceHandle superpixelFuzzyWeight = InvalidGraphResource;
    GraphResourceHandle superpixelFuzzyIndex  = InvalidGraphResource;
    GraphResourceHandle clusterVisibilityMask = InvalidGraphResource;
    GraphResourceHandle avgVisibility        = InvalidGraphResource;
    GraphResourceHandle clusterGatheredLightPoints = InvalidGraphResource;
    GraphResourceHandle clusterLightPointCounts    = InvalidGraphResource;
    GraphResourceHandle lightTreeNodes       = InvalidGraphResource;
    GraphResourceHandle lightTreeCompactToLeaf = InvalidGraphResource;
    GraphResourceHandle lightTreeClusterRoots  = InvalidGraphResource;
    GraphResourceHandle lightTreeSortKeys      = InvalidGraphResource;
    GraphResourceHandle lightTreeDispatchArgs  = InvalidGraphResource;
    GraphResourceHandle lightTreeNodeVisited   = InvalidGraphResource;
    GraphResourceHandle superpixelClusterHeap  = InvalidGraphResource;
};
