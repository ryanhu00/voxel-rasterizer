// Analytic scenes for synthetic ground-truth datasets.
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
    virtual bool contains(Vec3 p_world) const = 0;
    virtual Vec3 color_at(Vec3 p_world) const = 0;   // RGB in [0, 1]
    virtual Vec3 background() const { return Vec3{0.05f, 0.05f, 0.08f}; }
    virtual void bounds(Vec3& bmin, Vec3& bmax) const = 0;
};

std::unique_ptr<AnalyticScene> make_sphere_scene(Vec3 center, float radius);
std::unique_ptr<AnalyticScene> make_cube_scene(Vec3 center, Vec3 half_extents);
// Two spheres joined by a cylinder along X.
std::unique_ptr<AnalyticScene> make_dumbbell_scene(Vec3 center,
                                                   float sphere_radius,
                                                   float arm_radius,
                                                   float arm_length);

struct SynthOptions {
    int   width{256}, height{256};
    int   num_views{16};
    float orbit_radius{3.f};
    float orbit_elevation{0.3f};   // radians above equator
    float fov_y{0.9f};             // radians
    Vec3  look_target{0.f, 0.f, 0.f};
    Vec3  world_up{0.f, 1.f, 0.f};
    int   samples_per_ray{128};
};

// Writes rgb_NNNN.ppm, mask_NNNN.pgm, cameras.txt under `output_dir`.
bool synthesize_dataset(const AnalyticScene& scene,
                        const std::string&   output_dir,
                        const SynthOptions&  opts,
                        std::vector<CameraRecord>& out_records);

}  // namespace voxr
