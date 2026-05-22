#include "voxr/camera.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace voxr {

Camera Camera::from_look_at(int width, int height, float fov_y_radians,
                            Vec3 eye, Vec3 target, Vec3 up) {
    Camera c;
    c.width  = width;
    c.height = height;
    c.fy     = 0.5f * static_cast<float>(height) /
               std::tan(0.5f * fov_y_radians);
    c.fx     = c.fy;  // square pixels
    c.cx     = 0.5f * static_cast<float>(width);
    c.cy     = 0.5f * static_cast<float>(height);
    c.position = eye;
    c.R      = look_at_rotation(eye, target, up);
    return c;
}

Vec3 Camera::world_to_cam(Vec3 p_world) const {
    return R * (p_world - position);
}

bool Camera::project(Vec3 p_world, float& u, float& v, float& depth) const {
    Vec3 p_cam = world_to_cam(p_world);
    if (p_cam.z <= 1e-6f) return false;       // behind / on the camera plane
    u     = fx * p_cam.x / p_cam.z + cx;
    v     = fy * p_cam.y / p_cam.z + cy;
    depth = p_cam.z;
    return true;
}

void Camera::unproject_ray(float u, float v, Vec3& origin,
                           Vec3& direction) const {
    // Ray direction in camera frame, then rotate into world frame using R^T.
    Vec3 d_cam{(u - cx) / fx, (v - cy) / fy, 1.f};
    direction = normalize(transpose(R) * d_cam);
    origin    = position;
}

// ---- Dataset I/O ------------------------------------------------------------
bool save_camera_dataset(const std::string& path,
                         const std::vector<CameraRecord>& records) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "save_camera_dataset: cannot open " << path << "\n";
        return false;
    }
    f << "# voxel-rasterizer camera dataset v1\n";
    f << "# format: see include/voxr/camera.hpp\n";
    f << "count " << records.size() << "\n\n";

    for (std::size_t i = 0; i < records.size(); ++i) {
        const CameraRecord& rec = records[i];
        const Camera& c = rec.camera;
        f << "camera " << i << "\n";
        f << "image " << (rec.image_path.empty() ? "-" : rec.image_path) << "\n";
        f << "mask "  << (rec.mask_path.empty()  ? "-" : rec.mask_path)  << "\n";
        f << "size "  << c.width << " " << c.height << "\n";
        f << "K "     << c.fx << " " << c.fy << " " << c.cx << " " << c.cy
          << "\n";
        f << "t "     << c.position.x << " " << c.position.y << " "
                      << c.position.z << "\n";
        f << "R "     << c.R(0, 0) << " " << c.R(0, 1) << " " << c.R(0, 2) << "\n"
          << "       " << c.R(1, 0) << " " << c.R(1, 1) << " " << c.R(1, 2) << "\n"
          << "       " << c.R(2, 0) << " " << c.R(2, 1) << " " << c.R(2, 2) << "\n";
        f << "end\n\n";
    }
    return static_cast<bool>(f);
}

namespace {

bool expect_token(std::istream& f, const std::string& expected) {
    std::string tok;
    if (!(f >> tok)) return false;
    return tok == expected;
}

}  // namespace

bool load_camera_dataset(const std::string& path,
                         std::vector<CameraRecord>& out) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "load_camera_dataset: cannot open " << path << "\n";
        return false;
    }
    out.clear();

    auto skip_comments = [&]() {
        while (true) {
            int c = f.peek();
            if (c == '#') {
                std::string ignored;
                std::getline(f, ignored);
            } else if (std::isspace(c)) {
                f.get();
            } else {
                break;
            }
        }
    };

    skip_comments();
    if (!expect_token(f, "count")) {
        std::cerr << "load_camera_dataset: missing 'count' header\n";
        return false;
    }
    std::size_t n = 0;
    if (!(f >> n)) return false;
    out.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        skip_comments();
        std::string tok;
        if (!(f >> tok) || tok != "camera") {
            std::cerr << "load_camera_dataset: expected 'camera', got '" << tok
                      << "'\n";
            return false;
        }
        std::size_t idx;
        if (!(f >> idx)) return false;

        CameraRecord rec;
        Camera& c = rec.camera;

        // Required records, in fixed order to keep parsing simple.
        if (!expect_token(f, "image")) return false;
        f >> rec.image_path;
        if (rec.image_path == "-") rec.image_path.clear();
        if (!expect_token(f, "mask")) return false;
        f >> rec.mask_path;
        if (rec.mask_path == "-") rec.mask_path.clear();

        if (!expect_token(f, "size")) return false;
        f >> c.width >> c.height;

        if (!expect_token(f, "K")) return false;
        f >> c.fx >> c.fy >> c.cx >> c.cy;

        if (!expect_token(f, "t")) return false;
        f >> c.position.x >> c.position.y >> c.position.z;

        if (!expect_token(f, "R")) return false;
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 3; ++col)
                f >> c.R(r, col);

        if (!expect_token(f, "end")) return false;
        out.push_back(std::move(rec));
    }
    return true;
}

}  // namespace voxr
