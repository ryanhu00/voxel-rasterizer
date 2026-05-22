// CPU ray-march renderer for voxel grids.
//
// One logical thread per output pixel. Each thread traces a single ray with
// Amanatides & Woo 3D-DDA grid traversal and stops at the first occupied
// voxel. This structure is intentionally chosen to map 1:1 onto a CUDA pixel
// kernel in Week 2.
#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/math.hpp"
#include "voxr/voxel_grid.hpp"

namespace voxr {

struct RenderOptions {
    // Background color (RGB in [0, 1]) for rays that miss every voxel.
    Vec3 background{0.05f, 0.05f, 0.08f};

    // Simple Lambertian shading using a finite-difference normal estimate.
    bool shading{true};

    // Direction (in world coords) the analytical light source comes FROM.
    // Will be normalized internally.
    Vec3 light_direction{0.4f, 0.8f, -0.5f};

    // Ambient lighting weight; final = ambient + (1-ambient) * lambert.
    float ambient{0.25f};

    // Hard upper bound on grid steps per ray (safety net).
    int max_steps{4096};
};

// Renders `grid` from `camera` into `out` (allocated to camera.width/height,
// 3 channels). Returns false on degenerate input.
bool render_cpu(const VoxelGrid& grid,
                const Camera&    camera,
                ImageU8&         out,
                const RenderOptions& opts = {});

}  // namespace voxr
