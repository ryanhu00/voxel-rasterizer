// Shared setup, timing, and correctness helpers for CPU vs GPU benchmark tests.
#pragma once

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/scene.hpp"
#include "voxr/voxel_grid.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
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

struct GridParity {
    std::size_t total_voxels{0};
    std::size_t occ_diffs{0};
    double      occ_diff_frac{0.0};
    std::size_t color_mismatches{0};
    int         max_channel_diff{0};
};

struct ImageParity {
    std::size_t total_channels{0};
    std::size_t changed_channels{0};
    double      changed_frac{0.0};
    double      mean_abs_diff{0.0};
    int         max_diff{0};
};

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

inline std::string metrics_dir() {
    if (const char* v = std::getenv("VOXR_BENCH_METRICS_DIR"))
        return v;
    return "test_artifacts/metrics";
}

inline std::string metrics_json_path(const char* test_name) {
    return (fs::path(metrics_dir()) / (std::string(test_name) + ".json")).string();
}

// Aggregated results written to JSON and echoed on stdout for ctest visibility.
struct MetricsReport {
    std::string name;
    BenchConfig config;
    bool        pass{false};
    bool        cuda_available{false};

    std::optional<double> cpu_avg_ms;
    std::optional<double> gpu_avg_ms;
    std::optional<double> speedup;

    std::optional<long long> occupied_voxels;

    // Render GPU breakdown.
    std::optional<double> gpu_upload_ms;
    std::optional<double> gpu_kernel_avg_ms;
    std::optional<double> gpu_kernel_min_ms;
    std::optional<double> gpu_readback_avg_ms;
    std::optional<double> gpu_frame_avg_ms;

    // Pipeline per-stage (cpu / gpu).
    std::optional<double> cpu_reconstruct_ms;
    std::optional<double> cpu_render_ms;
    std::optional<double> cpu_total_ms;
    std::optional<double> gpu_reconstruct_ms;
    std::optional<double> gpu_render_ms;
    std::optional<double> gpu_total_ms;

    std::optional<GridParity>  grid_parity;
    std::optional<ImageParity> image_parity;
};

namespace detail {

inline void json_indent(std::string& out, int depth) {
    out.append(static_cast<std::size_t>(depth * 2), ' ');
}

inline void json_key(std::string& out, int depth, const char* key) {
    json_indent(out, depth);
    out += '"';
    out += key;
    out += "\": ";
}

inline void json_comma(std::string& out, bool& first) {
    if (!first) out += ',';
    first = false;
}

inline void json_number(std::string& out, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    out += buf;
}

inline void json_string(std::string& out, const char* s) {
    out += '"';
    for (const char* p = s; *p; ++p) {
        if (*p == '"' || *p == '\\') out += '\\';
        out += *p;
    }
    out += '"';
}

inline void json_bool(std::string& out, bool v) {
    out += v ? "true" : "false";
}

inline void json_u64(std::string& out, std::uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu",
                  static_cast<unsigned long long>(v));
    out += buf;
}

inline void write_config_json(std::string& out, int depth,
                              const BenchConfig& cfg) {
    json_key(out, depth, "config");
    out += "{\n";
    bool first = true;
    auto kv_i = [&](const char* k, int v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", v);
        out += buf;
        out += '\n';
    };
    kv_i("grid_n", cfg.grid_n);
    kv_i("num_views", cfg.num_views);
    kv_i("synth_res", cfg.synth_res);
    kv_i("render_w", cfg.render_w);
    kv_i("render_h", cfg.render_h);
    kv_i("warmup_iters", cfg.warmup_iters);
    kv_i("bench_iters", cfg.bench_iters);
    json_indent(out, depth);
    out += '}';
}

inline void write_grid_parity_json(std::string& out, int depth,
                                   const GridParity& p) {
    out += "{\n";
    bool first = true;
    auto kv_u = [&](const char* k, std::uint64_t v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        json_u64(out, v);
        out += '\n';
    };
    auto kv_f = [&](const char* k, double v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        json_number(out, v);
        out += '\n';
    };
    auto kv_i = [&](const char* k, int v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", v);
        out += buf;
        out += '\n';
    };
    kv_u("total_voxels", p.total_voxels);
    kv_u("occ_diffs", p.occ_diffs);
    kv_f("occ_diff_frac", p.occ_diff_frac);
    kv_u("color_mismatches", p.color_mismatches);
    kv_i("max_channel_diff", p.max_channel_diff);
    json_indent(out, depth);
    out += '}';
}

inline void write_image_parity_json(std::string& out, int depth,
                                    const ImageParity& p) {
    out += "{\n";
    bool first = true;
    auto kv_u = [&](const char* k, std::uint64_t v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        json_u64(out, v);
        out += '\n';
    };
    auto kv_f = [&](const char* k, double v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        json_number(out, v);
        out += '\n';
    };
    auto kv_i = [&](const char* k, int v) {
        json_comma(out, first);
        json_key(out, depth + 1, k);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", v);
        out += buf;
        out += '\n';
    };
    kv_u("total_channels", p.total_channels);
    kv_u("changed_channels", p.changed_channels);
    kv_f("changed_frac", p.changed_frac);
    kv_f("mean_abs_diff", p.mean_abs_diff);
    kv_i("max_diff", p.max_diff);
    json_indent(out, depth);
    out += '}';
}

}  // namespace detail

inline std::string metrics_to_json(const MetricsReport& r) {
    using namespace detail;
    std::string out;
    out += "{\n";
    bool first = true;

    auto kv_s = [&](const char* k, const char* v) {
        json_comma(out, first);
        json_key(out, 1, k);
        json_string(out, v);
        out += '\n';
    };
    auto kv_b = [&](const char* k, bool v) {
        json_comma(out, first);
        json_key(out, 1, k);
        json_bool(out, v);
        out += '\n';
    };
    auto kv_opt_i = [&](const char* k, const std::optional<long long>& v) {
        if (!v) return;
        json_comma(out, first);
        json_key(out, 1, k);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", *v);
        out += buf;
        out += '\n';
    };

    kv_s("test", r.name.c_str());
    kv_b("pass", r.pass);
    kv_b("cuda_available", r.cuda_available);

    json_comma(out, first);
    write_config_json(out, 1, r.config);
    out += ",\n";

    json_key(out, 1, "timing");
    out += "{\n";
    bool tfirst = true;
    auto tkv = [&](const char* k, const std::optional<double>& v) {
        if (!v) return;
        json_comma(out, tfirst);
        json_key(out, 2, k);
        json_number(out, *v);
        out += '\n';
    };
    tkv("cpu_avg_ms", r.cpu_avg_ms);
    tkv("gpu_avg_ms", r.gpu_avg_ms);
    tkv("speedup", r.speedup);
    tkv("gpu_upload_ms", r.gpu_upload_ms);
    tkv("gpu_kernel_avg_ms", r.gpu_kernel_avg_ms);
    tkv("gpu_kernel_min_ms", r.gpu_kernel_min_ms);
    tkv("gpu_readback_avg_ms", r.gpu_readback_avg_ms);
    tkv("gpu_frame_avg_ms", r.gpu_frame_avg_ms);
    tkv("cpu_reconstruct_ms", r.cpu_reconstruct_ms);
    tkv("cpu_render_ms", r.cpu_render_ms);
    tkv("cpu_total_ms", r.cpu_total_ms);
    tkv("gpu_reconstruct_ms", r.gpu_reconstruct_ms);
    tkv("gpu_render_ms", r.gpu_render_ms);
    tkv("gpu_total_ms", r.gpu_total_ms);
    json_indent(out, 1);
    out += '}';
    out += ",\n";

    json_key(out, 1, "correctness");
    out += "{\n";
    bool cfirst = true;
    if (r.grid_parity) {
        json_comma(out, cfirst);
        json_key(out, 2, "grid");
        write_grid_parity_json(out, 2, *r.grid_parity);
        out += '\n';
    }
    if (r.image_parity) {
        json_comma(out, cfirst);
        json_key(out, 2, "image");
        write_image_parity_json(out, 2, *r.image_parity);
        out += '\n';
    }
    json_indent(out, 1);
    out += '}';

    kv_opt_i("occupied_voxels", r.occupied_voxels);

    json_indent(out, 0);
    out += "}\n";
    return out;
}

inline bool save_metrics_json(const MetricsReport& r) {
    const std::string path = metrics_json_path(r.name.c_str());
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "error: cannot write metrics to %s\n",
                     path.c_str());
        return false;
    }
    f << metrics_to_json(r);
    return static_cast<bool>(f);
}

inline void emit_metrics_summary(const MetricsReport& r,
                                 const std::string& json_path) {
    std::printf("%s %s", r.pass ? "PASS" : "FAIL", r.name.c_str());
    if (r.cpu_total_ms && r.gpu_total_ms) {
        std::printf(" cpu_total=%.1fms gpu_total=%.1fms",
                    *r.cpu_total_ms, *r.gpu_total_ms);
    } else {
        if (r.cpu_avg_ms) std::printf(" cpu=%.1fms", *r.cpu_avg_ms);
        if (r.gpu_avg_ms) std::printf(" gpu=%.1fms", *r.gpu_avg_ms);
        if (r.gpu_kernel_avg_ms)
            std::printf(" gpu_kernel=%.2fms", *r.gpu_kernel_avg_ms);
    }
    if (r.speedup) std::printf(" speedup=%.1fx", *r.speedup);
    std::printf(" -> %s\n", json_path.c_str());
}

inline bool finalize_report(MetricsReport& r) {
    const std::string path = metrics_json_path(r.name.c_str());
    if (!save_metrics_json(r)) return false;
    emit_metrics_summary(r, path);
    std::fprintf(stderr, "metrics_json: %s\n", path.c_str());
    return true;
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

// ---- Correctness comparison helpers -----------------------------------------

inline GridParity compare_grids(const voxr::VoxelGrid& cpu,
                                const voxr::VoxelGrid& gpu,
                                int color_threshold = 2) {
    GridParity p;
    p.total_voxels = cpu.voxel_count();
    for (std::size_t i = 0; i < p.total_voxels; ++i) {
        if (cpu.occupancy[i] != gpu.occupancy[i]) ++p.occ_diffs;
        if (cpu.occupancy[i] && gpu.occupancy[i]) {
            int dr = std::abs((int)cpu.color_r[i] - (int)gpu.color_r[i]);
            int dg = std::abs((int)cpu.color_g[i] - (int)gpu.color_g[i]);
            int db = std::abs((int)cpu.color_b[i] - (int)gpu.color_b[i]);
            int mx = std::max({dr, dg, db});
            if (mx > color_threshold) ++p.color_mismatches;
            if (mx > p.max_channel_diff) p.max_channel_diff = mx;
        }
    }
    p.occ_diff_frac = p.total_voxels > 0
        ? static_cast<double>(p.occ_diffs) / static_cast<double>(p.total_voxels)
        : 0.0;
    return p;
}

inline ImageParity compare_images(const voxr::ImageU8& cpu,
                                  const voxr::ImageU8& gpu,
                                  int channel_threshold = 4) {
    ImageParity p;
    p.total_channels = cpu.data.size();
    long long sad = 0;
    for (std::size_t i = 0; i < p.total_channels; ++i) {
        int d = std::abs((int)cpu.data[i] - (int)gpu.data[i]);
        sad += d;
        if (d > channel_threshold) ++p.changed_channels;
        if (d > p.max_diff) p.max_diff = d;
    }
    p.changed_frac = p.total_channels > 0
        ? static_cast<double>(p.changed_channels) /
          static_cast<double>(p.total_channels)
        : 0.0;
    p.mean_abs_diff = p.total_channels > 0
        ? static_cast<double>(sad) / static_cast<double>(p.total_channels)
        : 0.0;
    return p;
}

// Thresholds matching the former test_cuda_parity tolerances.
constexpr double kMaxOccDiffFrac       = 0.001;  // 0.1% of voxels
constexpr int    kMaxColorMismatches   = 64;
constexpr double kMaxPixelChangedFrac  = 0.01;   // 1% of channels

inline bool check_grid_parity(const GridParity& p) {
    bool ok = true;
    std::fprintf(stderr, "correctness (grid):\n");
    std::fprintf(stderr, "  occ_diffs          %zu / %zu (%.4f%%)\n",
                 p.occ_diffs, p.total_voxels, 100.0 * p.occ_diff_frac);
    std::fprintf(stderr, "  color_mismatches   %zu  (max_ch_diff=%d)\n",
                 p.color_mismatches, p.max_channel_diff);
    if (p.occ_diff_frac > kMaxOccDiffFrac) {
        std::fprintf(stderr, "  FAIL: occ_diff_frac %.4f%% > %.4f%%\n",
                     100.0 * p.occ_diff_frac, 100.0 * kMaxOccDiffFrac);
        ok = false;
    }
    if (p.color_mismatches > static_cast<std::size_t>(kMaxColorMismatches)) {
        std::fprintf(stderr, "  FAIL: color_mismatches %zu > %d\n",
                     p.color_mismatches, kMaxColorMismatches);
        ok = false;
    }
    return ok;
}

inline bool check_image_parity(const ImageParity& p) {
    bool ok = true;
    std::fprintf(stderr, "correctness (image):\n");
    std::fprintf(stderr, "  changed_channels   %zu / %zu (%.4f%%)\n",
                 p.changed_channels, p.total_channels,
                 100.0 * p.changed_frac);
    std::fprintf(stderr, "  mean_abs_diff      %.4f  max_diff=%d\n",
                 p.mean_abs_diff, p.max_diff);
    if (p.changed_frac > kMaxPixelChangedFrac) {
        std::fprintf(stderr, "  FAIL: changed_frac %.4f%% > %.4f%%\n",
                     100.0 * p.changed_frac, 100.0 * kMaxPixelChangedFrac);
        ok = false;
    }
    return ok;
}

}  // namespace voxr_bench
