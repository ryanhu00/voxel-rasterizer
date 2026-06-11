// Benchmark: full pipeline reconstruct + render (CPU vs GPU wall time).
#include "bench_common.hpp"

#include "voxr/reconstruct_cpu.hpp"
#include "voxr/render_cpu.hpp"
#ifdef VOXR_WITH_CUDA
#include "voxr/reconstruct_cuda.hpp"
#include "voxr/render_cuda.hpp"
#endif

#include <vector>

namespace {

using clock = std::chrono::steady_clock;
using voxr_bench::BenchConfig;
using voxr_bench::SphereDataset;

struct PipelineTiming {
    double reconstruct_ms{0.0};
    double render_ms{0.0};
    double total_ms{0.0};
};

PipelineTiming time_pipeline_cpu(const BenchConfig& cfg,
                                 const SphereDataset& ds,
                                 const voxr::Camera& cam) {
    std::vector<double> recon_samples, render_samples, total_samples;
    recon_samples.reserve(static_cast<std::size_t>(cfg.bench_iters));
    render_samples.reserve(static_cast<std::size_t>(cfg.bench_iters));
    total_samples.reserve(static_cast<std::size_t>(cfg.bench_iters));

    voxr::ImageU8 out;

    for (int i = 0; i < cfg.warmup_iters + cfg.bench_iters; ++i) {
        const auto t_total = clock::now();

        voxr::VoxelGrid grid = voxr_bench::make_grid(cfg);
        const auto t_recon = clock::now();
        voxr::reconstruct_cpu(grid, ds.cameras, ds.masks, ds.images);
        const double recon_ms = voxr_bench::ms_since(t_recon);

        const auto t_render = clock::now();
        if (!voxr::render_cpu(grid, cam, out)) return {};
        const double render_ms = voxr_bench::ms_since(t_render);

        const double total_ms = voxr_bench::ms_since(t_total);
        if (i >= cfg.warmup_iters) {
            recon_samples.push_back(recon_ms);
            render_samples.push_back(render_ms);
            total_samples.push_back(total_ms);
        }
    }

    return {voxr_bench::average_ms(recon_samples),
            voxr_bench::average_ms(render_samples),
            voxr_bench::average_ms(total_samples)};
}

#ifdef VOXR_WITH_CUDA
PipelineTiming time_pipeline_gpu(const BenchConfig& cfg,
                                 const SphereDataset& ds,
                                 const voxr::Camera& cam) {
    std::vector<double> recon_samples, render_samples, total_samples;
    recon_samples.reserve(static_cast<std::size_t>(cfg.bench_iters));
    render_samples.reserve(static_cast<std::size_t>(cfg.bench_iters));
    total_samples.reserve(static_cast<std::size_t>(cfg.bench_iters));

    voxr::ImageU8 out;

    for (int i = 0; i < cfg.warmup_iters + cfg.bench_iters; ++i) {
        const auto t_total = clock::now();

        voxr::VoxelGrid grid = voxr_bench::make_grid(cfg);
        const auto t_recon = clock::now();
        voxr::reconstruct_cuda(grid, ds.cameras, ds.masks, ds.images);
        const double recon_ms = voxr_bench::ms_since(t_recon);

        voxr::CudaVoxelRenderer renderer(grid);
        const auto t_render = clock::now();
        if (!renderer.render(cam, out)) return {};
        const double render_ms = voxr_bench::ms_since(t_render);

        const double total_ms = voxr_bench::ms_since(t_total);
        if (i >= cfg.warmup_iters) {
            recon_samples.push_back(recon_ms);
            render_samples.push_back(render_ms);
            total_samples.push_back(total_ms);
        }
    }

    return {voxr_bench::average_ms(recon_samples),
            voxr_bench::average_ms(render_samples),
            voxr_bench::average_ms(total_samples)};
}
#endif

}  // namespace

int main() {
    const BenchConfig cfg = voxr_bench::load_config();
    voxr_bench::log_header("bench_pipeline");
    voxr_bench::log_config(cfg);

    SphereDataset ds;
    if (!voxr_bench::prepare_sphere_dataset(cfg, "test_artifacts/bench_pipe_ds",
                                            ds)) {
        std::fprintf(stderr, "error: failed to prepare benchmark dataset\n");
        return 1;
    }

    const voxr::Camera cam = voxr_bench::make_render_camera(cfg);

    const PipelineTiming cpu = time_pipeline_cpu(cfg, ds, cam);
    if (cpu.total_ms <= 0.0) {
        std::fprintf(stderr, "error: cpu pipeline failed\n");
        return 1;
    }

    std::fprintf(stderr, "cpu:\n");
    voxr_bench::log_metric("reconstruct_avg_ms", cpu.reconstruct_ms, "ms");
    voxr_bench::log_metric("render_avg_ms", cpu.render_ms, "ms");
    voxr_bench::log_metric("total_avg_ms", cpu.total_ms, "ms");

#ifdef VOXR_WITH_CUDA
    const PipelineTiming gpu = time_pipeline_gpu(cfg, ds, cam);
    if (gpu.total_ms <= 0.0) {
        std::fprintf(stderr, "error: gpu pipeline failed\n");
        return 1;
    }

    std::fprintf(stderr, "gpu:\n");
    voxr_bench::log_metric("reconstruct_avg_ms", gpu.reconstruct_ms, "ms");
    voxr_bench::log_metric("render_avg_ms", gpu.render_ms, "ms");
    voxr_bench::log_metric("total_avg_ms", gpu.total_ms, "ms");
    voxr_bench::log_speedup(cpu.total_ms, gpu.total_ms);
#else
    std::fprintf(stderr, "gpu: n/a (built without CUDA)\n");
#endif

    std::fprintf(stderr, "PASS\n");
    return 0;
}
