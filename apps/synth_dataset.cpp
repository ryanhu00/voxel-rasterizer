// synth_dataset: render an analytic scene from a ring of cameras and save the
// resulting RGB images, silhouette masks, and camera dataset to disk.
//
// Usage:
//   synth_dataset --out <dir> [--shape sphere|cube|dumbbell] [--views N]
//                 [--res W H] [--orbit R] [--elev RAD] [--fov RAD]

#include "voxr/camera.hpp"
#include "voxr/scene.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string out_dir = "data/synth";
    std::string shape   = "sphere";
    int   views         = 16;
    int   width         = 256;
    int   height        = 256;
    float orbit         = 3.0f;
    float elevation     = 0.3f;
    float fov_y         = 0.9f;
    int   spp           = 256;   // samples-per-ray for ground-truth rendering
};

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                std::cerr << "missing value for " << s << "\n";
                std::exit(1);
            }
        };
        if (s == "--out")        { need(1); a.out_dir   = argv[++i]; }
        else if (s == "--shape") { need(1); a.shape     = argv[++i]; }
        else if (s == "--views") { need(1); a.views     = std::atoi(argv[++i]); }
        else if (s == "--res")   { need(2); a.width     = std::atoi(argv[++i]);
                                            a.height    = std::atoi(argv[++i]); }
        else if (s == "--orbit") { need(1); a.orbit     = std::atof(argv[++i]); }
        else if (s == "--elev")  { need(1); a.elevation = std::atof(argv[++i]); }
        else if (s == "--fov")   { need(1); a.fov_y     = std::atof(argv[++i]); }
        else if (s == "--spp")   { need(1); a.spp       = std::atoi(argv[++i]); }
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: synth_dataset --out DIR [--shape sphere|cube|dumbbell]\n"
                         "                     [--views N] [--res W H] [--orbit R]\n"
                         "                     [--elev RAD] [--fov RAD] [--spp N]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << s << "\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 1;

    std::unique_ptr<voxr::AnalyticScene> scene;
    if (a.shape == "sphere") {
        scene = voxr::make_sphere_scene({0.f, 0.f, 0.f}, 1.0f);
    } else if (a.shape == "cube") {
        scene = voxr::make_cube_scene({0.f, 0.f, 0.f}, {0.8f, 0.8f, 0.8f});
    } else if (a.shape == "dumbbell") {
        scene = voxr::make_dumbbell_scene({0.f, 0.f, 0.f}, 0.55f, 0.2f, 1.6f);
    } else {
        std::cerr << "unknown shape: " << a.shape << "\n";
        return 1;
    }

    voxr::SynthOptions opts;
    opts.width             = a.width;
    opts.height            = a.height;
    opts.num_views         = a.views;
    opts.orbit_radius      = a.orbit;
    opts.orbit_elevation   = a.elevation;
    opts.fov_y             = a.fov_y;
    opts.samples_per_ray   = a.spp;

    std::vector<voxr::CameraRecord> recs;
    if (!voxr::synthesize_dataset(*scene, a.out_dir, opts, recs)) {
        std::cerr << "synthesize_dataset failed\n";
        return 1;
    }
    std::cout << "Wrote " << recs.size() << " views to " << a.out_dir << "\n";
    return 0;
}
