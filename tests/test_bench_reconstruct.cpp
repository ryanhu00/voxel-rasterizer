// Benchmark: shape-from-silhouette reconstruction (CPU vs GPU wall time).
#include "bench_common.hpp"

#include "voxr/reconstruct_cpu.hpp"
#ifdef VOXR_WITH_CUDA
#include "voxr/reconstruct_cuda.hpp"
#endif

#include <vector>

namespace {

using clock = std::chrono::steady_clock;
using voxr_bench::BenchConfig;
using voxr_bench::SphereDataset;

voxr::VoxelGrid fresh_grid(const BenchConfig& cfg) {
    return voxr_bench::make_grid(cfg);
}

double time_reconstruct_cpu(const BenchConfig& cfg,
                            const SphereDataset& ds) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(cfg.bench_iters));

    for (int i = 0; i < cfg.warmup_iters + cfg.bench_iters; ++i) {
        voxr::VoxelGrid g = fresh_grid(cfg);
        const auto t0 = clock::now();
        voxr::reconstruct_cpu(g, ds.cameras, ds.masks, ds.images);
        const double ms = voxr_bench::ms_since(t0);
        if (i >= cfg.warmup_iters) samples.push_back(ms);
    }
    return voxr_bench::average_ms(samples);
}

#ifdef VOXR_WITH_CUDA
double time_reconstruct_gpu(const BenchConfig& cfg,
                            const SphereDataset& ds) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(cfg.bench_iters));

    for (int i = 0; i < cfg.warmup_iters + cfg.bench_iters; ++i) {
        voxr::VoxelGrid g = fresh_grid(cfg);
        const auto t0 = clock::now();
        voxr::reconstruct_cuda(g, ds.cameras, ds.masks, ds.images);
        const double ms = voxr_bench::ms_since(t0);
        if (i >= cfg.warmup_iters) samples.push_back(ms);
    }
    return voxr_bench::average_ms(samples);
}
#endif

}  // namespace

int main() {
    const BenchConfig cfg = voxr_bench::load_config();
    voxr_bench::MetricsReport report;
    report.name   = "bench_reconstruct";
    report.config = cfg;
#ifdef VOXR_WITH_CUDA
    report.cuda_available = true;
#endif

    voxr_bench::log_header(report.name.c_str());
    voxr_bench::log_config(cfg);

    SphereDataset ds;
    if (!voxr_bench::prepare_sphere_dataset(cfg, "test_artifacts/bench_recon_ds",
                                            ds)) {
        std::fprintf(stderr, "error: failed to prepare benchmark dataset\n");
        return 1;
    }

    const double cpu_ms = time_reconstruct_cpu(cfg, ds);
    report.cpu_avg_ms = cpu_ms;
    voxr_bench::log_metric("cpu_avg_ms", cpu_ms, "ms");

#ifdef VOXR_WITH_CUDA
    const double gpu_ms = time_reconstruct_gpu(cfg, ds);
    report.gpu_avg_ms = gpu_ms;
    report.speedup    = cpu_ms / gpu_ms;
    voxr_bench::log_metric("gpu_avg_ms", gpu_ms, "ms");
    voxr_bench::log_speedup(cpu_ms, gpu_ms);

    voxr::VoxelGrid g_cpu = fresh_grid(cfg), g_gpu = fresh_grid(cfg);
    voxr::reconstruct_cpu(g_cpu, ds.cameras, ds.masks, ds.images);
    voxr::reconstruct_cuda(g_gpu, ds.cameras, ds.masks, ds.images);
    report.grid_parity = voxr_bench::compare_grids(g_cpu, g_gpu);
    if (!voxr_bench::check_grid_parity(*report.grid_parity)) {
        report.pass = false;
        voxr_bench::finalize_report(report);
        return 1;
    }
#else
    std::fprintf(stderr, "  gpu_avg_ms              n/a (built without CUDA)\n");
#endif

    report.pass = true;
    if (!voxr_bench::finalize_report(report)) return 1;
    return 0;
}
