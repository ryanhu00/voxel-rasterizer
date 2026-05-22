// render: render a saved voxel grid from a viewpoint of choice.
//
// Two modes:
//   1) Single image:   render --voxels grid.bin --out img.ppm
//                            [--eye X Y Z] [--target X Y Z] [--up X Y Z]
//                            [--res W H] [--fov RAD]
//   2) Orbit turntable: render --voxels grid.bin --orbit DIR --views N
//                              [--radius R] [--elev RAD] [--res W H]

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/render_cpu.hpp"
#include "voxr/voxel_grid.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string voxels_path;
    std::string out_path  = "render.ppm";
    std::string orbit_dir;
    int   orbit_views     = 0;
    float orbit_radius    = 3.0f;
    float orbit_elev      = 0.3f;

    int   width = 512, height = 512;
    float fov_y = 0.9f;

    float ex = 3.0f, ey = 1.0f, ez = 3.0f;
    float tx = 0.f,  ty = 0.f,  tz = 0.f;
    float ux = 0.f,  uy = 1.f,  uz = 0.f;
    bool  no_shading = false;
};

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) { std::cerr << "missing value for " << s
                                              << "\n"; std::exit(1); }
        };
        if (s == "--voxels")        { need(1); a.voxels_path = argv[++i]; }
        else if (s == "--out")      { need(1); a.out_path    = argv[++i]; }
        else if (s == "--orbit")    { need(1); a.orbit_dir   = argv[++i]; }
        else if (s == "--views")    { need(1); a.orbit_views = std::atoi(argv[++i]); }
        else if (s == "--radius")   { need(1); a.orbit_radius = std::atof(argv[++i]); }
        else if (s == "--elev")     { need(1); a.orbit_elev   = std::atof(argv[++i]); }
        else if (s == "--res")      { need(2); a.width = std::atoi(argv[++i]);
                                               a.height = std::atoi(argv[++i]); }
        else if (s == "--fov")      { need(1); a.fov_y = std::atof(argv[++i]); }
        else if (s == "--eye")      { need(3); a.ex = std::atof(argv[++i]);
                                               a.ey = std::atof(argv[++i]);
                                               a.ez = std::atof(argv[++i]); }
        else if (s == "--target")   { need(3); a.tx = std::atof(argv[++i]);
                                               a.ty = std::atof(argv[++i]);
                                               a.tz = std::atof(argv[++i]); }
        else if (s == "--up")       { need(3); a.ux = std::atof(argv[++i]);
                                               a.uy = std::atof(argv[++i]);
                                               a.uz = std::atof(argv[++i]); }
        else if (s == "--no-shading") { a.no_shading = true; }
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: render --voxels FILE [--out PATH | --orbit DIR --views N]\n"
                         "              [--res W H] [--fov RAD]\n"
                         "              [--eye X Y Z] [--target X Y Z] [--up X Y Z]\n"
                         "              [--radius R] [--elev RAD] [--no-shading]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << s << "\n"; return false;
        }
    }
    if (a.voxels_path.empty()) {
        std::cerr << "--voxels FILE is required\n"; return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 1;

    voxr::VoxelGrid grid;
    if (!voxr::load_voxel_grid(a.voxels_path, grid)) return 1;

    voxr::RenderOptions ropts;
    ropts.shading = !a.no_shading;

    if (!a.orbit_dir.empty() && a.orbit_views > 0) {
        std::error_code ec;
        fs::create_directories(a.orbit_dir, ec);
        const float two_pi = 6.28318530718f;
        for (int i = 0; i < a.orbit_views; ++i) {
            float phi = (i / static_cast<float>(a.orbit_views)) * two_pi;
            float h   = a.orbit_radius * std::sin(a.orbit_elev);
            float rho = a.orbit_radius * std::cos(a.orbit_elev);
            voxr::Vec3 eye{a.tx + rho * std::cos(phi),
                           a.ty + h,
                           a.tz + rho * std::sin(phi)};
            voxr::Camera cam = voxr::Camera::from_look_at(
                a.width, a.height, a.fov_y, eye,
                {a.tx, a.ty, a.tz}, {a.ux, a.uy, a.uz});
            voxr::ImageU8 img;
            if (!voxr::render_cpu(grid, cam, img, ropts)) return 1;
            char buf[256];
            std::snprintf(buf, sizeof(buf), "frame_%04d.ppm", i);
            voxr::save_ppm((fs::path(a.orbit_dir) / buf).string(), img);
        }
        std::cout << "Wrote " << a.orbit_views << " frames to "
                  << a.orbit_dir << "\n";
        return 0;
    }

    voxr::Camera cam = voxr::Camera::from_look_at(
        a.width, a.height, a.fov_y,
        {a.ex, a.ey, a.ez}, {a.tx, a.ty, a.tz}, {a.ux, a.uy, a.uz});
    voxr::ImageU8 img;
    if (!voxr::render_cpu(grid, cam, img, ropts)) return 1;
    if (!voxr::save_ppm(a.out_path, img)) return 1;
    std::cout << "Wrote " << a.out_path << "\n";
    return 0;
}
