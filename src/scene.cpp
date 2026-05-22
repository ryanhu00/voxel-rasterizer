#include "voxr/scene.hpp"

#include "voxr/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace voxr {
namespace fs = std::filesystem;

// ---- Concrete scenes ---------------------------------------------------------
namespace {

struct SphereScene : AnalyticScene {
    Vec3  center;
    float radius;
    SphereScene(Vec3 c, float r) : center(c), radius(r) {}
    bool contains(Vec3 p) const override {
        return length2(p - center) <= radius * radius;
    }
    Vec3 color_at(Vec3 p) const override {
        // Color by surface normal direction for an easy visual cue.
        Vec3 n = normalize(p - center);
        return Vec3{0.5f + 0.5f * n.x, 0.5f + 0.5f * n.y, 0.5f + 0.5f * n.z};
    }
    void bounds(Vec3& bmin, Vec3& bmax) const override {
        bmin = center - Vec3{radius, radius, radius};
        bmax = center + Vec3{radius, radius, radius};
    }
};

struct CubeScene : AnalyticScene {
    Vec3 center;
    Vec3 half;
    CubeScene(Vec3 c, Vec3 h) : center(c), half(h) {}
    bool contains(Vec3 p) const override {
        Vec3 d = p - center;
        return std::fabs(d.x) <= half.x && std::fabs(d.y) <= half.y &&
               std::fabs(d.z) <= half.z;
    }
    Vec3 color_at(Vec3 p) const override {
        Vec3 d = p - center;
        Vec3 a{std::fabs(d.x) / half.x,
               std::fabs(d.y) / half.y,
               std::fabs(d.z) / half.z};
        // Color picks the dominant face (cheap face-normal proxy).
        if (a.x >= a.y && a.x >= a.z)
            return d.x > 0 ? Vec3{0.9f, 0.3f, 0.3f} : Vec3{0.3f, 0.9f, 0.3f};
        if (a.y >= a.z)
            return d.y > 0 ? Vec3{0.3f, 0.3f, 0.9f} : Vec3{0.9f, 0.9f, 0.3f};
        return d.z > 0 ? Vec3{0.9f, 0.3f, 0.9f} : Vec3{0.3f, 0.9f, 0.9f};
    }
    void bounds(Vec3& bmin, Vec3& bmax) const override {
        bmin = center - half;
        bmax = center + half;
    }
};

struct DumbbellScene : AnalyticScene {
    Vec3  center;
    float sr;     // sphere radius
    float ar;     // arm radius
    float al;     // arm length (distance between sphere centers)
    DumbbellScene(Vec3 c, float sphere_r, float arm_r, float arm_l)
        : center(c), sr(sphere_r), ar(arm_r), al(arm_l) {}

    bool contains(Vec3 p) const override {
        Vec3 d = p - center;
        // Two spheres at +- al/2 along X.
        Vec3 dl{d.x + al * 0.5f, d.y, d.z};
        Vec3 dr{d.x - al * 0.5f, d.y, d.z};
        if (length2(dl) <= sr * sr) return true;
        if (length2(dr) <= sr * sr) return true;
        // Cylinder of radius ar along X between the two centers.
        if (std::fabs(d.x) <= al * 0.5f) {
            float r2 = d.y * d.y + d.z * d.z;
            if (r2 <= ar * ar) return true;
        }
        return false;
    }
    Vec3 color_at(Vec3 p) const override {
        Vec3 d = p - center;
        if (d.x < -al * 0.25f) return Vec3{0.95f, 0.55f, 0.25f};
        if (d.x >  al * 0.25f) return Vec3{0.25f, 0.55f, 0.95f};
        return Vec3{0.7f, 0.7f, 0.7f};
    }
    void bounds(Vec3& bmin, Vec3& bmax) const override {
        float ext_x = al * 0.5f + sr;
        float ext_y = std::max(sr, ar);
        float ext_z = ext_y;
        bmin = center - Vec3{ext_x, ext_y, ext_z};
        bmax = center + Vec3{ext_x, ext_y, ext_z};
    }
};

}  // namespace

std::unique_ptr<AnalyticScene> make_sphere_scene(Vec3 c, float r) {
    return std::make_unique<SphereScene>(c, r);
}
std::unique_ptr<AnalyticScene> make_cube_scene(Vec3 c, Vec3 h) {
    return std::make_unique<CubeScene>(c, h);
}
std::unique_ptr<AnalyticScene> make_dumbbell_scene(Vec3 c, float sr, float ar,
                                                   float al) {
    return std::make_unique<DumbbellScene>(c, sr, ar, al);
}

// ---- Ray-AABB intersection (slab method) ------------------------------------
// Returns true if the ray (origin + t * dir, t >= 0) intersects the box, and
// fills tmin/tmax with the entry / exit parameters (clamped at t >= 0).
static bool ray_aabb(Vec3 ro, Vec3 rd, Vec3 bmin, Vec3 bmax,
                     float& tmin, float& tmax) {
    float tmin_ = 0.f;
    float tmax_ = 1e30f;
    for (int axis = 0; axis < 3; ++axis) {
        float o = ro[axis], d = rd[axis];
        float lo = bmin[axis], hi = bmax[axis];
        if (std::fabs(d) < 1e-12f) {
            if (o < lo || o > hi) return false;
        } else {
            float inv = 1.f / d;
            float t1 = (lo - o) * inv;
            float t2 = (hi - o) * inv;
            if (t1 > t2) std::swap(t1, t2);
            tmin_ = std::max(tmin_, t1);
            tmax_ = std::min(tmax_, t2);
            if (tmin_ > tmax_) return false;
        }
    }
    tmin = tmin_;
    tmax = tmax_;
    return true;
}

// Single-ray, single-scene first-hit query by sampling along the ray.
// Returns true if a hit was found and stores the world-space hit position.
static bool first_hit_along_ray(const AnalyticScene& scene,
                                Vec3 ro, Vec3 rd, int max_samples,
                                Vec3& hit_position) {
    Vec3 bmin, bmax;
    scene.bounds(bmin, bmax);
    float tmin, tmax;
    if (!ray_aabb(ro, rd, bmin, bmax, tmin, tmax)) return false;
    if (tmax < 0.f) return false;
    tmin = std::max(tmin, 0.f);

    float t  = tmin;
    float dt = (tmax - tmin) / static_cast<float>(max_samples);
    if (dt <= 0.f) return false;
    for (int i = 0; i < max_samples; ++i) {
        Vec3 p = ro + rd * t;
        if (scene.contains(p)) {
            // Optional refinement: a couple of bisection steps for sharper
            // edges. The dataset is GROUND TRUTH, so cheap refinement here
            // pays off in better silhouettes.
            float a = t - dt, b = t;
            for (int it = 0; it < 6; ++it) {
                float m = 0.5f * (a + b);
                if (scene.contains(ro + rd * m)) b = m;
                else                              a = m;
            }
            hit_position = ro + rd * b;
            return true;
        }
        t += dt;
    }
    return false;
}

// ---- Dataset synthesis -------------------------------------------------------
bool synthesize_dataset(const AnalyticScene& scene,
                        const std::string&   output_dir,
                        const SynthOptions&  opts,
                        std::vector<CameraRecord>& out_records) {
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "synthesize_dataset: cannot create '" << output_dir
                  << "': " << ec.message() << "\n";
        return false;
    }

    out_records.clear();
    out_records.reserve(opts.num_views);

    const float two_pi = 6.28318530718f;

    for (int i = 0; i < opts.num_views; ++i) {
        // Place camera on a horizontal ring around the target.
        float phi = (static_cast<float>(i) / opts.num_views) * two_pi;
        float h   = opts.orbit_radius * std::sin(opts.orbit_elevation);
        float rho = opts.orbit_radius * std::cos(opts.orbit_elevation);
        Vec3  eye{opts.look_target.x + rho * std::cos(phi),
                  opts.look_target.y + h,
                  opts.look_target.z + rho * std::sin(phi)};
        Camera cam = Camera::from_look_at(opts.width, opts.height, opts.fov_y,
                                          eye, opts.look_target, opts.world_up);

        ImageU8 rgb(opts.width, opts.height, 3);
        ImageU8 mask(opts.width, opts.height, 1);

        Vec3 bg = scene.background();

        for (int y = 0; y < opts.height; ++y) {
            for (int x = 0; x < opts.width; ++x) {
                Vec3 ro, rd;
                cam.unproject_ray(static_cast<float>(x) + 0.5f,
                                  static_cast<float>(y) + 0.5f, ro, rd);
                Vec3 hit;
                bool h = first_hit_along_ray(scene, ro, rd,
                                             opts.samples_per_ray, hit);
                Vec3 col = h ? scene.color_at(hit) : bg;

                std::size_t idx = rgb.pixel_index(x, y);
                rgb.data[idx + 0] =
                    static_cast<std::uint8_t>(clampf(col.x, 0.f, 1.f) * 255.f);
                rgb.data[idx + 1] =
                    static_cast<std::uint8_t>(clampf(col.y, 0.f, 1.f) * 255.f);
                rgb.data[idx + 2] =
                    static_cast<std::uint8_t>(clampf(col.z, 0.f, 1.f) * 255.f);

                mask.data[mask.pixel_index(x, y)] = h ? 255 : 0;
            }
        }

        char buf[256];
        std::snprintf(buf, sizeof(buf), "rgb_%04d.ppm", i);
        std::string rgb_rel = buf;
        std::snprintf(buf, sizeof(buf), "mask_%04d.pgm", i);
        std::string mask_rel = buf;

        if (!save_ppm((fs::path(output_dir) / rgb_rel).string(), rgb))
            return false;
        if (!save_pgm((fs::path(output_dir) / mask_rel).string(), mask))
            return false;

        CameraRecord rec;
        rec.camera     = cam;
        rec.image_path = rgb_rel;
        rec.mask_path  = mask_rel;
        out_records.push_back(rec);
    }

    return save_camera_dataset((fs::path(output_dir) / "cameras.txt").string(),
                               out_records);
}

}  // namespace voxr
