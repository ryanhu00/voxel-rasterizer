#include "test_harness.hpp"

#include "voxr/voxel_grid.hpp"

#include <cstdio>
#include <filesystem>

int main() {
    using namespace voxr;
    namespace fs = std::filesystem;

    VoxelGrid g;
    g.resize(4, 5, 6);
    g.origin     = Vec3{-1.f, -2.f, -3.f};
    g.voxel_size = 0.25f;

    VOXR_EXPECT(g.voxel_count() == 4ull * 5 * 6);
    VOXR_EXPECT(g.linear_index(0, 0, 0) == 0);
    VOXR_EXPECT(g.linear_index(3, 0, 0) == 3);
    VOXR_EXPECT(g.linear_index(0, 1, 0) == 4);
    VOXR_EXPECT(g.linear_index(0, 0, 1) == 20);

    // voxel_center sanity
    Vec3 c = g.voxel_center(0, 0, 0);
    VOXR_EXPECT_NEAR(c.x, -1.f + 0.5f * 0.25f, 1e-6);
    VOXR_EXPECT_NEAR(c.y, -2.f + 0.5f * 0.25f, 1e-6);
    VOXR_EXPECT_NEAR(c.z, -3.f + 0.5f * 0.25f, 1e-6);

    // world_to_grid is the inverse offset.
    Vec3 vp = g.world_to_grid(c);
    VOXR_EXPECT_NEAR(vp.x, 0.5, 1e-6);
    VOXR_EXPECT_NEAR(vp.y, 0.5, 1e-6);
    VOXR_EXPECT_NEAR(vp.z, 0.5, 1e-6);

    // Set a few voxels, including color, then round-trip via disk.
    g.occupancy[g.linear_index(2, 3, 4)] = 1;
    g.color_r  [g.linear_index(2, 3, 4)] = 12;
    g.color_g  [g.linear_index(2, 3, 4)] = 34;
    g.color_b  [g.linear_index(2, 3, 4)] = 200;
    g.occupancy[g.linear_index(0, 0, 0)] = 1;

    fs::create_directories("test_artifacts");
    const std::string path = "test_artifacts/grid_roundtrip.bin";
    VOXR_EXPECT(save_voxel_grid(path, g));

    VoxelGrid h;
    VOXR_EXPECT(load_voxel_grid(path, h));
    VOXR_EXPECT(h.nx == g.nx && h.ny == g.ny && h.nz == g.nz);
    VOXR_EXPECT_NEAR(h.voxel_size, g.voxel_size, 1e-6);
    VOXR_EXPECT_NEAR(h.origin.x, g.origin.x, 1e-6);
    VOXR_EXPECT_NEAR(h.origin.y, g.origin.y, 1e-6);
    VOXR_EXPECT_NEAR(h.origin.z, g.origin.z, 1e-6);
    VOXR_EXPECT(h.occupied_count() == g.occupied_count());
    VOXR_EXPECT(h.color_r[h.linear_index(2, 3, 4)] == 12);
    VOXR_EXPECT(h.color_g[h.linear_index(2, 3, 4)] == 34);
    VOXR_EXPECT(h.color_b[h.linear_index(2, 3, 4)] == 200);

    VOXR_TEST_RETURN();
}
