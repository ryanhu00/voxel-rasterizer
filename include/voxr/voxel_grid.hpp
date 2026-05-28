// Uniform 3D grid: occupancy + RGB, SoA layout.
// Linear index = x + nx*(y + ny*z). Channels split for coalesced GPU loads.
// GPU plan: bind each channel as a 3D texture/surface for cached neighbour
// fetches during DDA and silhouette projection.
#pragma once

#include "voxr/math.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace voxr {

struct VoxelGrid {
    int   nx{0}, ny{0}, nz{0};
    Vec3  origin{0.f, 0.f, 0.f};   // world position of voxel (0,0,0) corner
    float voxel_size{1.f};

    std::vector<std::uint8_t> occupancy;
    std::vector<std::uint8_t> color_r;
    std::vector<std::uint8_t> color_g;
    std::vector<std::uint8_t> color_b;

    void resize(int nx_, int ny_, int nz_);
    std::size_t voxel_count() const {
        return static_cast<std::size_t>(nx) * ny * nz;
    }

    std::size_t linear_index(int x, int y, int z) const {
        return static_cast<std::size_t>(x) +
               static_cast<std::size_t>(nx) *
                   (static_cast<std::size_t>(y) +
                    static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
    }
    bool in_bounds(int x, int y, int z) const {
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    Vec3 voxel_center(int x, int y, int z) const {
        return Vec3{origin.x + (x + 0.5f) * voxel_size,
                    origin.y + (y + 0.5f) * voxel_size,
                    origin.z + (z + 0.5f) * voxel_size};
    }

    // AABB max corner (exclusive).
    Vec3 max_corner() const {
        return Vec3{origin.x + nx * voxel_size,
                    origin.y + ny * voxel_size,
                    origin.z + nz * voxel_size};
    }

    Vec3 world_to_grid(Vec3 p_world) const {
        return Vec3{(p_world.x - origin.x) / voxel_size,
                    (p_world.y - origin.y) / voxel_size,
                    (p_world.z - origin.z) / voxel_size};
    }

    std::size_t occupied_count() const;
};

// Binary format (little-endian):
//   "VOXG" | u32 version=1 | i32 nx,ny,nz | f32 origin[3] | f32 voxel_size
//   | u8 occupancy[N] | u8 color_r[N] | u8 color_g[N] | u8 color_b[N]
bool save_voxel_grid(const std::string& path, const VoxelGrid& grid);
bool load_voxel_grid(const std::string& path, VoxelGrid& out);

}  // namespace voxr
