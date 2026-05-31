// reconstruct: shape-from-silhouette voxel reconstruction (CPU).
//
// Usage:
//   reconstruct --in <dataset-dir> --out <voxels.bin>
//               [--grid N] [--bound F] [--center X Y Z]
//               [--min-views N] [--no-color]

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/reconstruct_cpu.hpp"
#include "voxr/voxel_grid.hpp"
#ifdef VOXR_WITH_CUDA
#include "voxr/reconstruct_cuda.hpp"
#endif

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string in_dir;
    std::string out_path = "voxels.bin";
    int   grid_n   = 128;
    float bound    = 1.2f;          // half-extent of cubic working volume
    float cx = 0.f, cy = 0.f, cz = 0.f;
    int   min_views = -1;
    bool  fuse_color = true;
    bool  gpu = false;
};

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                std::cerr << "missing value for " << s << "\n"; std::exit(1);
            }
        };
        if (s == "--in")             { need(1); a.in_dir    = argv[++i]; }
        else if (s == "--out")       { need(1); a.out_path  = argv[++i]; }
        else if (s == "--grid")      { need(1); a.grid_n    = std::atoi(argv[++i]); }
        else if (s == "--bound")     { need(1); a.bound     = std::atof(argv[++i]); }
        else if (s == "--center")    { need(3); a.cx = std::atof(argv[++i]);
                                                a.cy = std::atof(argv[++i]);
                                                a.cz = std::atof(argv[++i]); }
        else if (s == "--min-views") { need(1); a.min_views = std::atoi(argv[++i]); }
        else if (s == "--no-color")  { a.fuse_color = false; }
        else if (s == "--gpu")       { a.gpu = true; }
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: reconstruct --in DIR --out PATH [--grid N]\n"
                         "                   [--bound F] [--center X Y Z]\n"
                         "                   [--min-views N] [--no-color] [--gpu]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << s << "\n"; return false;
        }
    }
    if (a.in_dir.empty()) {
        std::cerr << "--in DIR is required\n"; return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 1;

    std::vector<voxr::CameraRecord> recs;
    fs::path cameras = fs::path(a.in_dir) / "cameras.txt";
    if (!voxr::load_camera_dataset(cameras.string(), recs)) {
        std::cerr << "failed to load " << cameras.string() << "\n";
        return 1;
    }

    std::vector<voxr::Camera>  cameras_vec;
    std::vector<voxr::ImageU8> masks;
    std::vector<voxr::ImageU8> images;
    cameras_vec.reserve(recs.size());
    masks.reserve(recs.size());
    images.reserve(recs.size());

    for (const auto& r : recs) {
        cameras_vec.push_back(r.camera);

        voxr::ImageU8 mask;
        if (r.mask_path.empty()) {
            std::cerr << "missing mask in dataset\n"; return 1;
        }
        if (!voxr::load_pnm((fs::path(a.in_dir) / r.mask_path).string(), mask)) {
            return 1;
        }
        masks.push_back(std::move(mask));

        if (a.fuse_color && !r.image_path.empty()) {
            voxr::ImageU8 img;
            if (!voxr::load_pnm((fs::path(a.in_dir) / r.image_path).string(),
                                img)) {
                return 1;
            }
            images.push_back(std::move(img));
        }
    }

    // Set up a cubic voxel volume centered at `--center`.
    voxr::VoxelGrid grid;
    grid.resize(a.grid_n, a.grid_n, a.grid_n);
    grid.origin     = voxr::Vec3{a.cx - a.bound, a.cy - a.bound, a.cz - a.bound};
    grid.voxel_size = 2.f * a.bound / a.grid_n;

    voxr::ReconstructOptions opts;
    opts.min_consistent_views = a.min_views;
    opts.fuse_color           = a.fuse_color;

    std::cout << "Reconstructing " << a.grid_n << "^3 voxels from "
              << recs.size() << " views (" << (a.gpu ? "GPU" : "CPU") << ")..."
              << std::endl;
    if (a.gpu) {
#ifdef VOXR_WITH_CUDA
        voxr::reconstruct_cuda(grid, cameras_vec, masks, images, opts);
#else
        std::cerr << "built without CUDA; rebuild with a CUDA toolkit\n";
        return 1;
#endif
    } else {
        voxr::reconstruct_cpu(grid, cameras_vec, masks, images, opts);
    }

    std::size_t occ = grid.occupied_count();
    std::cout << "Occupied voxels: " << occ << " ("
              << (100.0 * occ / grid.voxel_count()) << "% of volume)\n";

    if (!voxr::save_voxel_grid(a.out_path, grid)) return 1;
    std::cout << "Wrote " << a.out_path << "\n";
    return 0;
}
