// Benchmark: voxel rasterizer / ray-march (CPU vs GPU kernel time).
#include "bench_common.hpp"

#include "voxr/reconstruct_cpu.hpp"
#include "voxr/render_cpu.hpp"
#ifdef VOXR_WITH_CUDA
#include "voxr/render_cuda.hpp"
#endif

#include <vector>

namespace {

using clock = std::chrono::steady_clock;
using voxr_bench::BenchConfig;
using voxr_bench::SphereDataset;

bool build_reference_grid(const BenchConfig& cfg, const SphereDataset& ds,
                          voxr::VoxelGrid& grid) {
    grid = voxr_bench::make_grid(cfg);
    voxr::reconstruct_cpu(grid, ds.cameras, ds.masks, ds.images);
    return grid.occupied_count() > 0;
}

double time_render_cpu(const BenchConfig& cfg, const voxr::VoxelGrid& grid,
                       const voxr::Camera& cam) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(cfg.bench_iters));
    voxr::ImageU8 out;

    for (int i = 0; i < cfg.warmup_iters + cfg.bench_iters; ++i) {
        const auto t0 = clock::now();
        if (!voxr::render_cpu(grid, cam, out)) return -1.0;
        const double ms = voxr_bench::ms_since(t0);
        if (i >= cfg.warmup_iters) samples.push_back(ms);
    }
    return voxr_bench::average_ms(samples);
}

#ifdef VOXR_WITH_CUDA
struct GpuRenderTiming {
    double upload_ms{0.0};
    double kernel_avg_ms{0.0};
    double kernel_min_ms{0.0};
    double readback_avg_ms{0.0};
    double frame_avg_ms{0.0};
};

GpuRenderTiming time_render_gpu(const BenchConfig& cfg,
                                const voxr::VoxelGrid& grid,
                                const voxr::Camera& cam) {
    GpuRenderTiming t;
    voxr::CudaVoxelRenderer renderer(grid);
    t.upload_ms = renderer.upload_ms();

    voxr::ImageU8 out;
    std::vector<double> kernel, readback, frame;
    kernel.reserve(static_cast<std::size_t>(cfg.bench_iters));
    readback.reserve(static_cast<std::size_t>(cfg.bench_iters));
    frame.reserve(static_cast<std::size_t>(cfg.bench_iters));

    const int total = cfg.warmup_iters + cfg.bench_iters;
    for (int i = 0; i < total; ++i) {
        if (!renderer.render(cam, out)) return t;
        if (i >= cfg.warmup_iters) {
            kernel.push_back(renderer.last_kernel_ms());
            readback.push_back(renderer.last_readback_ms());
            frame.push_back(renderer.last_kernel_ms() +
                            renderer.last_readback_ms());
        }
    }

    t.kernel_avg_ms   = voxr_bench::average_ms(kernel);
    t.kernel_min_ms   = voxr_bench::min_ms(kernel);
    t.readback_avg_ms = voxr_bench::average_ms(readback);
    t.frame_avg_ms    = voxr_bench::average_ms(frame);
    return t;
}
#endif

}  // namespace

int main() {
    const BenchConfig cfg = voxr_bench::load_config();
    voxr_bench::log_header("bench_render");
    voxr_bench::log_config(cfg);

    SphereDataset ds;
    if (!voxr_bench::prepare_sphere_dataset(cfg, "test_artifacts/bench_render_ds",
                                            ds)) {
        std::fprintf(stderr, "error: failed to prepare benchmark dataset\n");
        return 1;
    }

    voxr::VoxelGrid grid;
    if (!build_reference_grid(cfg, ds, grid)) {
        std::fprintf(stderr, "error: reference grid is empty\n");
        return 1;
    }
    voxr_bench::log_metric_i("occupied_voxels",
                             static_cast<long long>(grid.occupied_count()));

    const voxr::Camera cam = voxr_bench::make_render_camera(cfg);

    const double cpu_ms = time_render_cpu(cfg, grid, cam);
    if (cpu_ms < 0.0) {
        std::fprintf(stderr, "error: render_cpu failed\n");
        return 1;
    }
    voxr_bench::log_metric("cpu_avg_ms", cpu_ms, "ms");

#ifdef VOXR_WITH_CUDA
    const GpuRenderTiming gpu = time_render_gpu(cfg, grid, cam);
    voxr_bench::log_metric("gpu_upload_ms", gpu.upload_ms, "ms  (once)");
    voxr_bench::log_metric("gpu_kernel_avg_ms", gpu.kernel_avg_ms, "ms");
    voxr_bench::log_metric("gpu_kernel_min_ms", gpu.kernel_min_ms, "ms");
    voxr_bench::log_metric("gpu_readback_avg_ms", gpu.readback_avg_ms, "ms");
    voxr_bench::log_metric("gpu_frame_avg_ms", gpu.frame_avg_ms,
                           "ms  (kernel+readback)");
    voxr_bench::log_speedup(cpu_ms, gpu.kernel_avg_ms);
#else
    std::fprintf(stderr, "  gpu_*                   n/a (built without CUDA)\n");
#endif

    std::fprintf(stderr, "PASS\n");
    return 0;
}
