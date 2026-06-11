// Shared setup and timing helpers for CPU vs GPU benchmark tests.
#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/scene.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace voxr_bench {

namespace fs = std::filesystem;

struct BenchConfig {
    int grid_n       = 128;
    int num_views    = 24;
    int synth_res    = 128;
    int render_w     = 512;
    int render_h     = 512;
    int warmup_iters = 1;
    int bench_iters  = 3;
};

struct TimedRun {
    double ms{0.0};
    bool   ok{false};
};

inline BenchConfig load_config() {
    BenchConfig c;
    if (const char* v = std::getenv("VOXR_BENCH_GRID"))
        c.grid_n = std::atoi(v);
    if (const char* v = std::getenv("VOXR_BENCH_VIEWS"))
        c.num_views = std::atoi(v);
    if (const char* v = std::getenv("VOXR_BENCH_SYNTH_RES"))
        c.synth_res = std::atoi(v);
    if (const char* v = std::getenv("VOXR_BENCH_RENDER_W"))
        c.render_w = std::atoi(v);
    if (const char* v = std::getenv("VOXR_BENCH_RENDER_H"))
        c.render_h = std::atoi(v);
    if (const char* v = std::getenv("VOXR_BENCH_WARMUP"))
        c.warmup_iters = std::atoi(v);
    if (const char* v = std::getenv("VOXR_BENCH_ITERS"))
        c.bench_iters = std::atoi(v);
    if (c.warmup_iters < 0) c.warmup_iters = 0;
    if (c.bench_iters < 1)  c.bench_iters = 1;
    return c;
}

inline double ms_since(const std::chrono::steady_clock::time_point& t0) {
    using ms = std::chrono::duration<double, std::milli>;
    return ms(std::chrono::steady_clock::now() - t0).count();
}

struct SphereDataset {
    std::string              dir;
    std::vector<voxr::Camera>  cameras;
    std::vector<voxr::ImageU8> masks;
    std::vector<voxr::ImageU8> images;
};

inline bool prepare_sphere_dataset(const BenchConfig& cfg,
                                   const std::string& dir,
                                   SphereDataset& out) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    auto scene = voxr::make_sphere_scene({0.f, 0.f, 0.f}, 0.8f);
    voxr::SynthOptions opts;
    opts.width           = cfg.synth_res;
    opts.height          = cfg.synth_res;
    opts.num_views       = cfg.num_views;
    opts.orbit_radius    = 3.f;
    opts.orbit_elevation = 0.3f;
    opts.fov_y           = 0.9f;
    opts.samples_per_ray = cfg.synth_res;

    std::vector<voxr::CameraRecord> recs;
    if (!voxr::synthesize_dataset(*scene, dir, opts, recs)) return false;

    out.dir = dir;
    out.cameras.clear();
    out.masks.clear();
    out.images.clear();
    out.cameras.reserve(recs.size());
    out.masks.reserve(recs.size());
    out.images.reserve(recs.size());

    for (const auto& r : recs) {
        out.cameras.push_back(r.camera);
        voxr::ImageU8 mask, img;
        if (!voxr::load_pnm((fs::path(dir) / r.mask_path).string(), mask))
            return false;
        if (!voxr::load_pnm((fs::path(dir) / r.image_path).string(), img))
            return false;
        out.masks.push_back(std::move(mask));
        out.images.push_back(std::move(img));
    }
    return true;
}

inline voxr::VoxelGrid make_grid(const BenchConfig& cfg) {
    voxr::VoxelGrid g;
    g.resize(cfg.grid_n, cfg.grid_n, cfg.grid_n);
    g.origin     = voxr::Vec3{-1.2f, -1.2f, -1.2f};
    g.voxel_size = 2.4f / cfg.grid_n;
    return g;
}

inline voxr::Camera make_render_camera(const BenchConfig& cfg) {
    return voxr::Camera::from_look_at(
        cfg.render_w, cfg.render_h, 0.9f,
        voxr::Vec3{2.2f, 1.2f, 2.2f},
        voxr::Vec3{0.f, 0.f, 0.f},
        voxr::Vec3{0.f, 1.f, 0.f});
}

inline void log_header(const char* name) {
    std::fprintf(stderr, "\n=== %s ===\n", name);
}

inline void log_config(const BenchConfig& cfg) {
    std::fprintf(stderr,
                 "config: grid=%d^3 views=%d synth=%dx%d render=%dx%d "
                 "warmup=%d iters=%d\n",
                 cfg.grid_n, cfg.num_views, cfg.synth_res, cfg.synth_res,
                 cfg.render_w, cfg.render_h, cfg.warmup_iters, cfg.bench_iters);
}

inline void log_metric(const char* key, double value, const char* unit) {
    std::fprintf(stderr, "  %-18s %10.3f %s\n", key, value, unit);
}

inline void log_metric_i(const char* key, long long value) {
    std::fprintf(stderr, "  %-18s %10lld\n", key, value);
}

inline void log_speedup(double cpu_ms, double gpu_ms) {
    if (gpu_ms > 0.0) {
        std::fprintf(stderr, "  %-18s %10.2fx\n", "speedup (cpu/gpu)",
                     cpu_ms / gpu_ms);
    } else {
        std::fprintf(stderr, "  speedup (cpu/gpu)      n/a (gpu not run)\n");
    }
}

inline double average_ms(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    double sum = 0.0;
    for (double s : samples) sum += s;
    return sum / static_cast<double>(samples.size());
}

inline double min_ms(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    double m = samples[0];
    for (double s : samples) m = std::min(m, s);
    return m;
}

}  // namespace voxr_bench
