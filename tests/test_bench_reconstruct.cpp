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
    voxr::VoxelGrid g = voxr_bench::make_grid(cfg);
    return g;
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
    voxr_bench::log_header("bench_reconstruct");
    voxr_bench::log_config(cfg);

    SphereDataset ds;
    if (!voxr_bench::prepare_sphere_dataset(cfg, "test_artifacts/bench_recon_ds",
                                            ds)) {
        std::fprintf(stderr, "error: failed to prepare benchmark dataset\n");
        return 1;
    }

    const double cpu_ms = time_reconstruct_cpu(cfg, ds);
    voxr_bench::log_metric("cpu_avg_ms", cpu_ms, "ms");

#ifdef VOXR_WITH_CUDA
    const double gpu_ms = time_reconstruct_gpu(cfg, ds);
    voxr_bench::log_metric("gpu_avg_ms", gpu_ms, "ms");
    voxr_bench::log_speedup(cpu_ms, gpu_ms);
#else
    std::fprintf(stderr, "  gpu_avg_ms              n/a (built without CUDA)\n");
#endif

    std::fprintf(stderr, "PASS\n");
    return 0;
}
