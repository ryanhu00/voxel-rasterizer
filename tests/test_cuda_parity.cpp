// GPU-vs-CPU parity. The CPU paths in voxr_core are the oracle: the CUDA
// kernels must reproduce occupancy and rendered pixels within tight bounds.
#include "test_harness.hpp"

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/reconstruct_cpu.hpp"
#include "voxr/reconstruct_cuda.hpp"
#include "voxr/render_cpu.hpp"
#include "voxr/render_cuda.hpp"
#include "voxr/scene.hpp"
#include "voxr/voxel_grid.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    using namespace voxr;

    // ---- Synthesize a small sphere dataset (same as test_pipeline) -----------
    auto scene = make_sphere_scene({0.f, 0.f, 0.f}, 0.6f);
    SynthOptions sopts;
    sopts.width = 96; sopts.height = 96; sopts.num_views = 12;
    sopts.orbit_radius = 2.5f; sopts.orbit_elevation = 0.4f;
    sopts.fov_y = 0.9f; sopts.samples_per_ray = 96;

    const std::string ds_dir = "test_artifacts/parity_ds";
    std::error_code ec; fs::remove_all(ds_dir, ec);

    std::vector<CameraRecord> recs;
    VOXR_EXPECT(synthesize_dataset(*scene, ds_dir, sopts, recs));

    std::vector<Camera>  cams;
    std::vector<ImageU8> masks, images;
    for (const auto& r : recs) {
        cams.push_back(r.camera);
        ImageU8 m, img;
        VOXR_EXPECT(load_pnm((fs::path(ds_dir) / r.mask_path).string(), m));
        VOXR_EXPECT(load_pnm((fs::path(ds_dir) / r.image_path).string(), img));
        masks.push_back(std::move(m));
        images.push_back(std::move(img));
    }

    auto make_grid = [] {
        VoxelGrid g;
        g.resize(64, 64, 64);
        g.origin = Vec3{-0.8f, -0.8f, -0.8f};
        g.voxel_size = 1.6f / 64.f;
        return g;
    };

    // ---- Reconstruction parity -----------------------------------------------
    VoxelGrid g_cpu = make_grid(), g_gpu = make_grid();
    reconstruct_cpu(g_cpu, cams, masks, images);
    reconstruct_cuda(g_gpu, cams, masks, images);

    std::size_t occ_diff = 0, color_diff = 0;
    const std::size_t N = g_cpu.voxel_count();
    for (std::size_t i = 0; i < N; ++i) {
        if (g_cpu.occupancy[i] != g_gpu.occupancy[i]) ++occ_diff;
        // Color only matters where both agree the voxel is occupied.
        if (g_cpu.occupancy[i] && g_gpu.occupancy[i]) {
            int dr = std::abs((int)g_cpu.color_r[i] - (int)g_gpu.color_r[i]);
            int dg = std::abs((int)g_cpu.color_g[i] - (int)g_gpu.color_g[i]);
            int db = std::abs((int)g_cpu.color_b[i] - (int)g_gpu.color_b[i]);
            if (dr > 2 || dg > 2 || db > 2) ++color_diff;
        }
    }
    double occ_frac = (double)occ_diff / (double)N;
    std::fprintf(stderr, "recon: occ_diff=%zu (%.4f%%) color_diff=%zu\n",
                 occ_diff, 100.0 * occ_frac, color_diff);
    // A handful of boundary voxels may flip due to float-order differences.
    VOXR_EXPECT(occ_frac < 0.001);
    VOXR_EXPECT(color_diff < 64);

    // ---- Render parity --------------------------------------------------------
    // Render the CPU grid with both renderers so any difference is the renderer.
    Camera novel = Camera::from_look_at(96, 96, 0.9f, Vec3{1.5f, 1.0f, 1.5f},
                                        Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 1.f, 0.f});
    ImageU8 r_cpu, r_gpu;
    VOXR_EXPECT(render_cpu(g_cpu, novel, r_cpu));
    VOXR_EXPECT(render_cuda(g_cpu, novel, r_gpu));

    std::size_t px_diff = 0;
    long sad = 0;
    const std::size_t P = (std::size_t)r_cpu.width * r_cpu.height * 3;
    for (std::size_t i = 0; i < P; ++i) {
        int d = std::abs((int)r_cpu.data[i] - (int)r_gpu.data[i]);
        sad += d;
        if (d > 4) ++px_diff;
    }
    double px_frac = (double)px_diff / (double)P;
    std::fprintf(stderr, "render: changed_channels=%zu (%.4f%%) mean_abs=%.4f\n",
                 px_diff, 100.0 * px_frac, (double)sad / (double)P);
    VOXR_EXPECT(px_frac < 0.01);

    VOXR_TEST_RETURN();
}
