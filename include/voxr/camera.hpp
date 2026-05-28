// Pinhole camera, OpenCV conventions.
//   x_cam = R * (X_world - position);  [u v 1]^T ~ K * x_cam / z_cam
//   Camera frame: +X right, +Y down, +Z forward.
// GPU plan: small POD, lives in __constant__ memory (broadcast read).
#pragma once

#include "voxr/math.hpp"

#include <string>
#include <vector>

namespace voxr {

struct Camera {
    int   width{0}, height{0};
    float fx{0.f}, fy{0.f}, cx{0.f}, cy{0.f};
    Vec3  position{0.f, 0.f, 0.f};
    Mat3  R{Mat3::identity()};

    static Camera from_look_at(int width, int height,
                               float fov_y_radians,
                               Vec3 eye, Vec3 target, Vec3 up);

    Vec3 world_to_cam(Vec3 p_world) const;

    // False if behind camera (z_cam <= 0).
    bool project(Vec3 p_world, float& u, float& v, float& depth) const;

    // `direction` is unit length, `origin` = camera center.
    void unproject_ray(float u, float v, Vec3& origin, Vec3& direction) const;
};

// Textual dataset format, one record per camera:
//   camera <index>
//   image <path> | mask <path-or-"-">       (both optional)
//   size <w> <h>
//   K <fx> <fy> <cx> <cy>
//   t <tx> <ty> <tz>
//   R <r00..r22>                            (3 rows)
//   end
struct CameraRecord {
    Camera      camera;
    std::string image_path;
    std::string mask_path;
};

bool save_camera_dataset(const std::string& path,
                         const std::vector<CameraRecord>& records);
bool load_camera_dataset(const std::string& path,
                         std::vector<CameraRecord>& out);

}  // namespace voxr
