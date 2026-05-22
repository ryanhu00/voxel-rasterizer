#include "test_harness.hpp"

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/reconstruct_cpu.hpp"
#include "voxr/render_cpu.hpp"
#include "voxr/scene.hpp"
#include "voxr/voxel_grid.hpp"

#include <cstdio>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    using namespace voxr;

    // ---- 1) Synthesize a small sphere dataset --------------------------------
    auto scene = make_sphere_scene({0.f, 0.f, 0.f}, 0.6f);

    SynthOptions sopts;
    sopts.width            = 96;
    sopts.height           = 96;
    sopts.num_views        = 12;
    sopts.orbit_radius     = 2.5f;
    sopts.orbit_elevation  = 0.4f;
    sopts.fov_y            = 0.9f;
    sopts.samples_per_ray  = 96;

    const std::string ds_dir = "test_artifacts/sphere_ds";
    std::error_code ec;
    fs::remove_all(ds_dir, ec);

    std::vector<CameraRecord> recs;
    VOXR_EXPECT(synthesize_dataset(*scene, ds_dir, sopts, recs));
    VOXR_EXPECT(recs.size() == static_cast<std::size_t>(sopts.num_views));

    // ---- 2) Load masks/images and reconstruct --------------------------------
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

    VoxelGrid grid;
    grid.resize(64, 64, 64);
    grid.origin     = Vec3{-0.8f, -0.8f, -0.8f};
    grid.voxel_size = 1.6f / 64.f;

    reconstruct_cpu(grid, cams, masks, images);

    std::size_t occ = grid.occupied_count();
    VOXR_EXPECT(occ > 0);

    // ---- 3) Compare against analytic ground truth ----------------------------
    // We expect a sphere of radius 0.6 -> volume ≈ 0.905 cubic units inside a
    // 1.6^3 = 4.096 cubic unit volume, i.e. ~22% of voxels should be occupied.
    // SfS over-estimates (visual hull), so an upper-bounded check is fragile;
    // we instead validate that:
    //   - every truly-interior voxel was kept (high recall),
    //   - the over-estimation isn't catastrophic.
    std::size_t inside_gt = 0;
    std::size_t inside_kept = 0;
    std::size_t outside_kept = 0;
    for (int z = 0; z < grid.nz; ++z) {
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                Vec3 c = grid.voxel_center(x, y, z);
                bool in_gt = scene->contains(c);
                bool kept  = grid.occupancy[grid.linear_index(x, y, z)] != 0;
                if (in_gt) {
                    ++inside_gt;
                    if (kept) ++inside_kept;
                } else if (kept) {
                    ++outside_kept;
                }
            }
        }
    }
    double recall = static_cast<double>(inside_kept) /
                    static_cast<double>(inside_gt);
    double extra  = static_cast<double>(outside_kept) /
                    static_cast<double>(inside_gt);
    std::fprintf(stderr, "GT inside=%zu  kept_inside=%zu  kept_outside=%zu  "
                         "recall=%.3f  extra/gt=%.3f\n",
                 inside_gt, inside_kept, outside_kept, recall, extra);

    VOXR_EXPECT(recall > 0.95);   // we shouldn't carve away the true interior
    VOXR_EXPECT(extra  < 0.40);   // visual hull stays close to the sphere

    // ---- 4) Render and verify the output is non-trivial ----------------------
    RenderOptions ropts;
    Camera novel = Camera::from_look_at(96, 96, 0.9f,
                                        Vec3{1.5f, 1.0f, 1.5f},
                                        Vec3{0.f, 0.f, 0.f},
                                        Vec3{0.f, 1.f, 0.f});
    ImageU8 rendered;
    VOXR_EXPECT(render_cpu(grid, novel, rendered, ropts));

    std::size_t fg = 0;
    for (int i = 0; i < rendered.width * rendered.height; ++i) {
        std::uint8_t r = rendered.data[i * 3 + 0];
        std::uint8_t g = rendered.data[i * 3 + 1];
        std::uint8_t b = rendered.data[i * 3 + 2];
        // Background is dark blue ~(13, 13, 20); anything noticeably brighter
        // is treated as a foreground pixel.
        if (r > 40 || g > 40 || b > 40) ++fg;
    }
    double fg_frac = static_cast<double>(fg) /
                     static_cast<double>(rendered.width * rendered.height);
    std::fprintf(stderr, "rendered foreground fraction = %.3f\n", fg_frac);
    VOXR_EXPECT(fg_frac > 0.05 && fg_frac < 0.95);

    save_ppm("test_artifacts/sphere_render.ppm", rendered);

    VOXR_TEST_RETURN();
}
