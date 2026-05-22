// Analytic synthetic scenes used to generate ground-truth data for the
// reconstruction pipeline.
//
// Each scene answers two queries:
//   - contains(p)      : is the world-space point inside the object?
//   - color_at(p)      : color (RGB in [0,1]) the surface point would carry.
//
// For Week 1 these are enough to render silhouette masks (one occupancy
// query along each ray) and color images (closest-hit query). The
// formulation also lets us evaluate reconstruction quality voxel-by-voxel
// against ground truth.
#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/math.hpp"
#include "voxr/voxel_grid.hpp"

#include <memory>
#include <string>
#include <vector>

namespace voxr {

struct AnalyticScene {
    virtual ~AnalyticScene() = default;

    // True if `p_world` is inside the object.
    virtual bool contains(Vec3 p_world) const = 0;

    // Surface color at `p_world` (assumed near the surface). Components in
    // [0, 1].
    virtual Vec3 color_at(Vec3 p_world) const = 0;

    // Background color used when a ray misses the object.
    virtual Vec3 background() const { return Vec3{0.05f, 0.05f, 0.08f}; }

    // AABB containing the object; used to size voxel grids automatically.
    virtual void bounds(Vec3& bmin, Vec3& bmax) const = 0;
};

// ---- Concrete scenes ---------------------------------------------------------
std::unique_ptr<AnalyticScene> make_sphere_scene(Vec3 center, float radius);
std::unique_ptr<AnalyticScene> make_cube_scene(Vec3 center, Vec3 half_extents);
// Two spheres joined by a cylinder along the X axis.
std::unique_ptr<AnalyticScene> make_dumbbell_scene(Vec3 center,
                                                   float sphere_radius,
                                                   float arm_radius,
                                                   float arm_length);

// ---- Synthetic dataset generation -------------------------------------------
struct SynthOptions {
    int   width{256};
    int   height{256};
    int   num_views{16};         // distributed on a horizontal ring
    float orbit_radius{3.f};     // distance of cameras from `look_target`
    float orbit_elevation{0.3f}; // elevation (in radians) above the equator
    float fov_y{0.9f};           // vertical FOV in radians
    Vec3  look_target{0.f, 0.f, 0.f};
    Vec3  world_up{0.f, 1.f, 0.f};
    int   samples_per_ray{128};  // ray-marching steps for ground-truth render
};

// Generate `num_views` calibrated views of `scene` and write them under
// `output_dir`. Produces:
//   - rgb_NNNN.ppm     color rendering
//   - mask_NNNN.pgm    silhouette (255 = object, 0 = background)
//   - cameras.txt      dataset descriptor (see camera.hpp)
//
// Returns the camera records used (also written to cameras.txt).
bool synthesize_dataset(const AnalyticScene& scene,
                        const std::string&   output_dir,
                        const SynthOptions&  opts,
                        std::vector<CameraRecord>& out_records);

}  // namespace voxr
