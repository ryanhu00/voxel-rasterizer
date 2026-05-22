// CPU shape-from-silhouette reconstruction.
//
// Mirrors the structure the CUDA kernel will take in Week 2:
//   - One thread per voxel.
//   - The thread loops over all cameras.
//   - Each iteration: project voxel center, read silhouette, accumulate.
//
// The CPU version uses a parallel-for over voxel z-slices (cheap and trivial)
// so that swapping in `<<<grid, block>>>(voxels...)` later is a near-1:1
// translation.
#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/voxel_grid.hpp"

#include <vector>

namespace voxr {

struct ReconstructOptions {
    // Number of cameras (out of N) for which the voxel must lie inside the
    // silhouette in order to be marked occupied. Set < num_cameras for some
    // robustness to silhouette noise; defaults to "all".
    int min_consistent_views{-1};   // -1 => require all views

    // Whether to fuse RGB colors from `images`. If false, only occupancy is
    // computed and color_r/g/b are left zeroed.
    bool fuse_color{true};

    // Threshold used to classify silhouette pixels as foreground.
    std::uint8_t silhouette_threshold{128};
};

// Inputs:
//   - `grid` already sized (resize() called) with origin/voxel_size set.
//   - `cameras[i]` corresponds to mask `masks[i]` and (optionally) image
//     `images[i]`. Color fusion ignores entries where `images.size() < i`.
// Outputs:
//   - `grid.occupancy` and `grid.color_*` filled in.
void reconstruct_cpu(VoxelGrid&                       grid,
                     const std::vector<Camera>&       cameras,
                     const std::vector<ImageU8>&      masks,
                     const std::vector<ImageU8>&      images,
                     const ReconstructOptions&        opts = {});

}  // namespace voxr
