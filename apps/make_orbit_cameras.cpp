// make_orbit_cameras: generate a cameras.txt for a directory of images that
// were taken at evenly-spaced angles around an object on a horizontal ring.
//
// Designed for the "I have my own photos" path: drop converted PPMs and
// silhouette PGMs into a directory, run this once to write cameras.txt, then
// feed the directory straight into `reconstruct`.
//
// File naming is convention-based: by default the tool picks up `rgb_*.ppm`
// and `mask_*.pgm`, sorts each list alphabetically, and pairs them by index.
//
// Usage:
//   make_orbit_cameras --in DIR
//                      [--rgb GLOB] [--masks GLOB]
//                      [--radius R] [--elev DEG] [--fov RAD]
//                      [--target X Y Z] [--up X Y Z]
//                      [--start-deg D] [--cw]
//                      [--res W H] [--out PATH]

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string in_dir;
    std::string rgb_glob   = "rgb_*.ppm";
    std::string mask_glob  = "mask_*.pgm";
    std::string out_path;             // default: in_dir/cameras.txt
    int   width = 0, height = 0;      // 0 = infer from first image
    float fov_y    = 0.9f;
    float radius   = 3.f;
    float elev_deg = 0.f;
    float start_deg = 0.f;
    bool  cw = false;
    float tx = 0.f, ty = 0.f, tz = 0.f;
    float ux = 0.f, uy = 1.f, uz = 0.f;
};

// Minimal glob matcher supporting `*` (any run, including empty) and `?` (any
// single char). Enough for filename patterns; we deliberately avoid pulling in
// a regex dep.
bool glob_match(const std::string& pat, const std::string& s) {
    size_t i = 0, j = 0;
    size_t star = std::string::npos, mark = 0;
    while (j < s.size()) {
        if (i < pat.size() && (pat[i] == '?' || pat[i] == s[j])) {
            ++i; ++j;
        } else if (i < pat.size() && pat[i] == '*') {
            star = i++;
            mark = j;
        } else if (star != std::string::npos) {
            i = star + 1;
            j = ++mark;
        } else {
            return false;
        }
    }
    while (i < pat.size() && pat[i] == '*') ++i;
    return i == pat.size();
}

std::vector<std::string> list_matching(const fs::path& dir,
                                       const std::string& glob) {
    std::vector<std::string> out;
    if (glob.empty() || glob == "-") return out;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().string();
        if (glob_match(glob, name)) out.push_back(name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                std::cerr << "missing value for " << s << "\n"; std::exit(1);
            }
        };
        if      (s == "--in")        { need(1); a.in_dir    = argv[++i]; }
        else if (s == "--rgb")       { need(1); a.rgb_glob  = argv[++i]; }
        else if (s == "--masks")     { need(1); a.mask_glob = argv[++i]; }
        else if (s == "--out")       { need(1); a.out_path  = argv[++i]; }
        else if (s == "--radius")    { need(1); a.radius    = std::atof(argv[++i]); }
        else if (s == "--elev")      { need(1); a.elev_deg  = std::atof(argv[++i]); }
        else if (s == "--fov")       { need(1); a.fov_y     = std::atof(argv[++i]); }
        else if (s == "--start-deg") { need(1); a.start_deg = std::atof(argv[++i]); }
        else if (s == "--cw")        { a.cw = true; }
        else if (s == "--res")       { need(2); a.width  = std::atoi(argv[++i]);
                                                a.height = std::atoi(argv[++i]); }
        else if (s == "--target")    { need(3); a.tx = std::atof(argv[++i]);
                                                a.ty = std::atof(argv[++i]);
                                                a.tz = std::atof(argv[++i]); }
        else if (s == "--up")        { need(3); a.ux = std::atof(argv[++i]);
                                                a.uy = std::atof(argv[++i]);
                                                a.uz = std::atof(argv[++i]); }
        else if (s == "-h" || s == "--help") {
            std::cout <<
              "Usage: make_orbit_cameras --in DIR\n"
              "                          [--rgb GLOB] [--masks GLOB]\n"
              "                          [--radius R] [--elev DEG] [--fov RAD]\n"
              "                          [--target X Y Z] [--up X Y Z]\n"
              "                          [--start-deg D] [--cw]\n"
              "                          [--res W H] [--out PATH]\n"
              "\n"
              "Pairs the i-th matching --rgb file with the i-th matching --masks\n"
              "file (alphabetical order) and writes a ring-orbit cameras.txt.\n"
              "Use --masks '' to skip mask pairing (reconstruct will then fail;\n"
              "useful for cameras-only use cases).\n";
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

    fs::path dir = a.in_dir;
    if (!fs::is_directory(dir)) {
        std::cerr << "not a directory: " << dir << "\n"; return 1;
    }

    auto rgbs  = list_matching(dir, a.rgb_glob);
    auto masks = list_matching(dir, a.mask_glob);

    if (rgbs.empty()) {
        std::cerr << "no files matching --rgb '" << a.rgb_glob
                  << "' under " << dir << "\n";
        return 1;
    }
    if (!a.mask_glob.empty() && a.mask_glob != "-" &&
        masks.size() != rgbs.size()) {
        std::cerr << "warning: " << rgbs.size() << " rgb files but "
                  << masks.size() << " mask files; pairing by index up to min."
                  << "\n";
    }

    // Infer resolution from the first PPM if not explicitly given.
    if (a.width <= 0 || a.height <= 0) {
        voxr::ImageU8 probe;
        if (!voxr::load_pnm((dir / rgbs[0]).string(), probe)) {
            std::cerr << "failed to read " << (dir / rgbs[0]) << " to infer size\n";
            return 1;
        }
        a.width  = probe.width;
        a.height = probe.height;
    }

    const float deg2rad = 3.14159265358979f / 180.f;
    const float two_pi  = 6.28318530718f;
    const float elev    = a.elev_deg  * deg2rad;
    const float start   = a.start_deg * deg2rad;

    const int n = static_cast<int>(rgbs.size());
    std::vector<voxr::CameraRecord> recs;
    recs.reserve(n);
    for (int i = 0; i < n; ++i) {
        float t   = static_cast<float>(i) / n;
        float phi = start + (a.cw ? -t : t) * two_pi;
        float h   = a.radius * std::sin(elev);
        float rho = a.radius * std::cos(elev);
        voxr::Vec3 eye{a.tx + rho * std::cos(phi),
                       a.ty + h,
                       a.tz + rho * std::sin(phi)};
        voxr::Camera cam = voxr::Camera::from_look_at(
            a.width, a.height, a.fov_y,
            eye, {a.tx, a.ty, a.tz}, {a.ux, a.uy, a.uz});

        voxr::CameraRecord rec;
        rec.camera     = cam;
        rec.image_path = rgbs[i];
        if (i < static_cast<int>(masks.size())) rec.mask_path = masks[i];
        recs.push_back(rec);
    }

    fs::path out = a.out_path.empty() ? (dir / "cameras.txt") : fs::path(a.out_path);
    if (!voxr::save_camera_dataset(out.string(), recs)) {
        std::cerr << "failed to write " << out << "\n"; return 1;
    }

    std::cout << "Wrote " << recs.size() << " cameras to " << out.string()
              << " (" << a.width << "x" << a.height
              << ", radius=" << a.radius
              << ", elev=" << a.elev_deg << " deg"
              << ", fov_y=" << a.fov_y << " rad"
              << ", " << (a.cw ? "CW" : "CCW") << ")\n";
    return 0;
}
