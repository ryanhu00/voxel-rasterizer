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
#ifdef VOXR_WITH_CUDA
#include "voxr/render_cuda.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
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
    bool  gpu = false;
    int   bench = 0;       // >0: render N frames, report GPU timings, exit
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
        else if (s == "--gpu")        { a.gpu = true; }
        else if (s == "--bench")      { need(1); a.bench = std::atoi(argv[++i]);
                                                 a.gpu = true; }
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: render --voxels FILE [--out PATH | --orbit DIR --views N]\n"
                         "              [--res W H] [--fov RAD]\n"
                         "              [--eye X Y Z] [--target X Y Z] [--up X Y Z]\n"
                         "              [--radius R] [--elev RAD] [--no-shading] [--gpu]\n"
                         "              [--bench N]   (resident GPU loop, report timings)\n";
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

// Dispatch to GPU or CPU renderer.
bool render_dispatch(bool gpu, const voxr::VoxelGrid& grid,
                     const voxr::Camera& cam, voxr::ImageU8& img,
                     const voxr::RenderOptions& ropts) {
    if (gpu) {
#ifdef VOXR_WITH_CUDA
        return voxr::render_cuda(grid, cam, img, ropts);
#else
        std::cerr << "built without CUDA; rebuild with a CUDA toolkit\n";
        return false;
#endif
    }
    return voxr::render_cpu(grid, cam, img, ropts);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 1;

    voxr::VoxelGrid grid;
    if (!voxr::load_voxel_grid(a.voxels_path, grid)) return 1;

    voxr::RenderOptions ropts;
    ropts.shading = !a.no_shading;

    // ---- Benchmark: grid-resident GPU loop, report isolated timings ---------
    if (a.bench > 0) {
#ifdef VOXR_WITH_CUDA
        voxr::Camera cam = voxr::Camera::from_look_at(
            a.width, a.height, a.fov_y,
            {a.ex, a.ey, a.ez}, {a.tx, a.ty, a.tz}, {a.ux, a.uy, a.uz});
        voxr::CudaVoxelRenderer renderer(grid);  // one-time upload
        voxr::ImageU8 img;
        float k_sum = 0.f, rb_sum = 0.f, k_min = 1e30f;
        for (int i = 0; i < a.bench; ++i) {
            if (!renderer.render(cam, img, ropts)) return 1;
            k_sum  += renderer.last_kernel_ms();
            rb_sum += renderer.last_readback_ms();
            k_min   = std::min(k_min, renderer.last_kernel_ms());
        }
        float k_avg = k_sum / a.bench, rb_avg = rb_sum / a.bench;
        std::printf("bench: %dx%d, %d^3 grid, %d frames\n",
                    a.width, a.height, grid.nx, a.bench);
        std::printf("  grid upload (1x) : %8.3f ms\n", renderer.upload_ms());
        std::printf("  kernel    avg/min: %8.3f / %.3f ms  (%.1f / %.1f fps)\n",
                    k_avg, k_min, 1000.f / k_avg, 1000.f / k_min);
        std::printf("  readback  avg    : %8.3f ms\n", rb_avg);
        std::printf("  frame (kernel+rb): %8.3f ms  (%.1f fps)\n",
                    k_avg + rb_avg, 1000.f / (k_avg + rb_avg));
        return 0;
#else
        std::cerr << "--bench needs a CUDA build\n";
        return 1;
#endif
    }

    if (!a.orbit_dir.empty() && a.orbit_views > 0) {
        std::error_code ec;
        fs::create_directories(a.orbit_dir, ec);
        const float two_pi = 6.28318530718f;
#ifdef VOXR_WITH_CUDA
        // Grid-resident GPU loop: upload once, render every frame on-device.
        std::unique_ptr<voxr::CudaVoxelRenderer> gpu_renderer;
        if (a.gpu) gpu_renderer = std::make_unique<voxr::CudaVoxelRenderer>(grid);
#endif
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
#ifdef VOXR_WITH_CUDA
            if (gpu_renderer) {
                if (!gpu_renderer->render(cam, img, ropts)) return 1;
            } else
#endif
            if (!render_dispatch(a.gpu, grid, cam, img, ropts)) return 1;
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
    if (!render_dispatch(a.gpu, grid, cam, img, ropts)) return 1;
    if (!voxr::save_ppm(a.out_path, img)) return 1;
    std::cout << "Wrote " << a.out_path << "\n";
    return 0;
}
