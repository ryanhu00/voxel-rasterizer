// VoxelGrid: a uniform 3D grid of occupancy + RGB.
//
// Layout: structure-of-arrays, linear index = x + nx * (y + ny * z).
// We store occupancy as uint8 (0/1) and color as 3 separate uint8 channels.
// Keeping channels split rather than interleaved makes it easier to perform
// coalesced loads from CUDA kernels in later weeks.
#pragma once

#include "voxr/math.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace voxr {

struct VoxelGrid {
    // Grid dimensions (number of voxels along each axis).
    int nx{0};
    int ny{0};
    int nz{0};

    // World-space position of voxel (0,0,0) corner and uniform voxel size.
    Vec3  origin{0.f, 0.f, 0.f};
    float voxel_size{1.f};

    std::vector<std::uint8_t> occupancy;   // size = nx*ny*nz
    std::vector<std::uint8_t> color_r;     // size = nx*ny*nz
    std::vector<std::uint8_t> color_g;
    std::vector<std::uint8_t> color_b;

    // ---- Construction --------------------------------------------------------
    void resize(int nx_, int ny_, int nz_);
    std::size_t voxel_count() const {
        return static_cast<std::size_t>(nx) * ny * nz;
    }

    // ---- Indexing helpers ----------------------------------------------------
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

    // World coordinates of the grid AABB (min corner inclusive, max exclusive).
    Vec3 max_corner() const {
        return Vec3{origin.x + nx * voxel_size,
                    origin.y + ny * voxel_size,
                    origin.z + nz * voxel_size};
    }

    // Convert a world-space position to floating voxel coordinates (no
    // clipping). Useful for ray traversal initialization.
    Vec3 world_to_grid(Vec3 p_world) const {
        return Vec3{(p_world.x - origin.x) / voxel_size,
                    (p_world.y - origin.y) / voxel_size,
                    (p_world.z - origin.z) / voxel_size};
    }

    // ---- Statistics ----------------------------------------------------------
    std::size_t occupied_count() const;
};

// ---- Serialization -----------------------------------------------------------
// Binary format:
//   magic       : 4 bytes "VOXG"
//   version     : uint32 = 1
//   nx, ny, nz  : int32 x 3
//   origin      : float32 x 3
//   voxel_size  : float32
//   occupancy   : nx*ny*nz uint8
//   color_r     : nx*ny*nz uint8
//   color_g     : nx*ny*nz uint8
//   color_b     : nx*ny*nz uint8
//
// All numbers are little-endian (host byte order on the platforms we target).
bool save_voxel_grid(const std::string& path, const VoxelGrid& grid);
bool load_voxel_grid(const std::string& path, VoxelGrid& out);

}  // namespace voxr
