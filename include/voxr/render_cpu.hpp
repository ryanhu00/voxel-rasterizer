// CPU voxel-grid renderer (Amanatides-Woo 3D-DDA, one thread per pixel).
// GPU plan: 1 thread per pixel, 16x16 tiles for ray coherence. Occupancy/color
// as 3D textures; camera in __constant__ mem. DDA loop is the divergence
// hotspot — keep max_steps tight, escalate to persistent threads if needed.
#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/math.hpp"
#include "voxr/voxel_grid.hpp"

namespace voxr {

struct RenderOptions {
    Vec3  background{0.05f, 0.05f, 0.08f};
    bool  shading{true};                          // Lambertian on face normal
    Vec3  light_direction{0.4f, 0.8f, -0.5f};     // normalized internally
    float ambient{0.25f};                         // final = ambient + (1-a)*lambert
    int   max_steps{4096};
};

// Resizes `out` to camera.width x camera.height x 3 if needed.
bool render_cpu(const VoxelGrid& grid,
                const Camera&    camera,
                ImageU8&         out,
                const RenderOptions& opts = {});

}  // namespace voxr
