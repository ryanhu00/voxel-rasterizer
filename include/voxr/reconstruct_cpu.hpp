// CPU shape-from-silhouette reconstruction.
// GPU plan: 1 thread per voxel, 8x8x8 blocks. Cameras in __constant__ mem,
// masks/images as 2D textures (free hardware bilinear). Writes are coalesced
// and atomic-free since each voxel has a unique owner.

#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/voxel_grid.hpp"

#include <vector>

namespace voxr {

struct ReconstructOptions {
    int          min_consistent_views{-1};  // -1 => all views
    bool         fuse_color{true};
    std::uint8_t silhouette_threshold{128};
};

// Requires `grid` pre-sized; `cameras[i]` pairs with `masks[i]` and optional
// `images[i]`. Fills grid.occupancy and grid.color_*.
void reconstruct_cpu(VoxelGrid&                       grid,
                     const std::vector<Camera>&       cameras,
                     const std::vector<ImageU8>&      masks,
                     const std::vector<ImageU8>&      images,
                     const ReconstructOptions&        opts = {});

}  // namespace voxr
