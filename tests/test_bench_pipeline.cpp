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
    voxr_bench::MetricsReport report;
    report.name   = "bench_pipeline";
    report.config = cfg;
#ifdef VOXR_WITH_CUDA
    report.cuda_available = true;
#endif

    voxr_bench::log_header(report.name.c_str());
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
    report.cpu_reconstruct_ms = cpu.reconstruct_ms;
    report.cpu_render_ms      = cpu.render_ms;
    report.cpu_total_ms       = cpu.total_ms;

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
    report.gpu_reconstruct_ms = gpu.reconstruct_ms;
    report.gpu_render_ms      = gpu.render_ms;
    report.gpu_total_ms       = gpu.total_ms;
    report.speedup            = cpu.total_ms / gpu.total_ms;

    std::fprintf(stderr, "gpu:\n");
    voxr_bench::log_metric("reconstruct_avg_ms", gpu.reconstruct_ms, "ms");
    voxr_bench::log_metric("render_avg_ms", gpu.render_ms, "ms");
    voxr_bench::log_metric("total_avg_ms", gpu.total_ms, "ms");
    voxr_bench::log_speedup(cpu.total_ms, gpu.total_ms);

    voxr::VoxelGrid g_cpu = voxr_bench::make_grid(cfg);
    voxr::VoxelGrid g_gpu = voxr_bench::make_grid(cfg);
    voxr::reconstruct_cpu(g_cpu, ds.cameras, ds.masks, ds.images);
    voxr::reconstruct_cuda(g_gpu, ds.cameras, ds.masks, ds.images);

    report.grid_parity = voxr_bench::compare_grids(g_cpu, g_gpu);
    if (!voxr_bench::check_grid_parity(*report.grid_parity)) {
        report.pass = false;
        voxr_bench::finalize_report(report);
        return 1;
    }

    voxr::ImageU8 img_cpu, img_gpu;
    voxr::render_cpu(g_cpu, cam, img_cpu);
    voxr::render_cuda(g_cpu, cam, img_gpu);

    report.image_parity = voxr_bench::compare_images(img_cpu, img_gpu);
    if (!voxr_bench::check_image_parity(*report.image_parity)) {
        report.pass = false;
        voxr_bench::finalize_report(report);
        return 1;
    }
#else
    std::fprintf(stderr, "gpu: n/a (built without CUDA)\n");
#endif

    report.pass = true;
    if (!voxr_bench::finalize_report(report)) return 1;
    return 0;
}
