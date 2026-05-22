#include "voxr/voxel_grid.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace voxr {

void VoxelGrid::resize(int nx_, int ny_, int nz_) {
    nx = nx_;
    ny = ny_;
    nz = nz_;
    const std::size_t n = voxel_count();
    occupancy.assign(n, 0);
    color_r.assign(n, 0);
    color_g.assign(n, 0);
    color_b.assign(n, 0);
}

std::size_t VoxelGrid::occupied_count() const {
    std::size_t count = 0;
    for (std::uint8_t v : occupancy) count += (v != 0);
    return count;
}

namespace {

constexpr char kMagic[4]    = {'V', 'O', 'X', 'G'};
constexpr std::uint32_t kVer = 1;

template <typename T>
void write_pod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool read_pod(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.gcount() == static_cast<std::streamsize>(sizeof(T));
}

}  // namespace

bool save_voxel_grid(const std::string& path, const VoxelGrid& grid) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "save_voxel_grid: cannot open " << path << "\n";
        return false;
    }
    f.write(kMagic, 4);
    write_pod(f, kVer);
    std::int32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    write_pod(f, nx); write_pod(f, ny); write_pod(f, nz);
    write_pod(f, grid.origin.x); write_pod(f, grid.origin.y);
    write_pod(f, grid.origin.z);
    write_pod(f, grid.voxel_size);

    const std::streamsize n = static_cast<std::streamsize>(grid.voxel_count());
    if (static_cast<std::size_t>(n) != grid.occupancy.size()) {
        std::cerr << "save_voxel_grid: inconsistent occupancy size\n";
        return false;
    }
    f.write(reinterpret_cast<const char*>(grid.occupancy.data()), n);
    f.write(reinterpret_cast<const char*>(grid.color_r.data()),   n);
    f.write(reinterpret_cast<const char*>(grid.color_g.data()),   n);
    f.write(reinterpret_cast<const char*>(grid.color_b.data()),   n);
    return static_cast<bool>(f);
}

bool load_voxel_grid(const std::string& path, VoxelGrid& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "load_voxel_grid: cannot open " << path << "\n";
        return false;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0) {
        std::cerr << "load_voxel_grid: bad magic in " << path << "\n";
        return false;
    }
    std::uint32_t ver = 0;
    if (!read_pod(f, ver) || ver != kVer) {
        std::cerr << "load_voxel_grid: unsupported version " << ver << "\n";
        return false;
    }
    std::int32_t nx, ny, nz;
    if (!read_pod(f, nx) || !read_pod(f, ny) || !read_pod(f, nz)) return false;

    Vec3 origin;
    float voxel_size = 0.f;
    if (!read_pod(f, origin.x) || !read_pod(f, origin.y) ||
        !read_pod(f, origin.z) || !read_pod(f, voxel_size)) {
        return false;
    }

    out.resize(nx, ny, nz);
    out.origin     = origin;
    out.voxel_size = voxel_size;

    const std::streamsize n = static_cast<std::streamsize>(out.voxel_count());
    f.read(reinterpret_cast<char*>(out.occupancy.data()), n);
    f.read(reinterpret_cast<char*>(out.color_r.data()),   n);
    f.read(reinterpret_cast<char*>(out.color_g.data()),   n);
    f.read(reinterpret_cast<char*>(out.color_b.data()),   n);
    return static_cast<bool>(f);
}

}  // namespace voxr
