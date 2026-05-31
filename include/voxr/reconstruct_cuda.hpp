// CUDA shape-from-silhouette reconstruction. Drop-in for reconstruct_cpu:
// one thread per voxel, masks/images in global memory, manual bilinear.
#pragma once

#include "voxr/reconstruct_cpu.hpp"  // ReconstructOptions, types

namespace voxr {

// Same contract as reconstruct_cpu. Fills grid.occupancy and grid.color_*.
void reconstruct_cuda(VoxelGrid&                  grid,
                      const std::vector<Camera>&  cameras,
                      const std::vector<ImageU8>& masks,
                      const std::vector<ImageU8>& images,
                      const ReconstructOptions&   opts = {});

}  // namespace voxr
