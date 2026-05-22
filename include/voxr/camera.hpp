// Pinhole camera with OpenCV-style intrinsics and extrinsics.
//
// World -> Camera:  x_cam = R * (X_world - t_world)        where t_world is the
//                                                          camera center in world
// World -> Pixel :  [u v 1]^T ~ K * x_cam   (after divide by z_cam)
//
// Camera frame: +X right, +Y down, +Z forward (into the scene).
#pragma once

#include "voxr/math.hpp"

#include <string>
#include <vector>

namespace voxr {

struct Camera {
    // Intrinsics
    int   width{0};
    int   height{0};
    float fx{0.f};
    float fy{0.f};
    float cx{0.f};
    float cy{0.f};

    // Extrinsics: position of the camera center in world coords, plus rotation
    // R from world to camera frame.
    Vec3 position{0.f, 0.f, 0.f};
    Mat3 R{Mat3::identity()};

    // ---- Construction helpers ------------------------------------------------
    static Camera from_look_at(int width, int height,
                               float fov_y_radians,
                               Vec3 eye, Vec3 target, Vec3 up);

    // ---- Geometry ------------------------------------------------------------
    // Transform a world point into the camera frame.
    Vec3 world_to_cam(Vec3 p_world) const;

    // Project a world point to pixel coords. Returns false if the point is
    // behind the camera (z_cam <= 0). On success, fills `u`, `v`, and `depth`
    // (which equals z_cam, NOT the Euclidean distance to the point).
    bool project(Vec3 p_world, float& u, float& v, float& depth) const;

    // Backproject pixel coordinates into a world-space ray. The returned
    // direction is unit length, and the origin equals the camera center.
    void unproject_ray(float u, float v, Vec3& origin, Vec3& direction) const;
};

// ---- Camera dataset I/O ------------------------------------------------------
// Simple textual format, one camera per record:
//
//   camera <index>
//   image  <path-relative-to-dataset-dir>
//   mask   <path-or-"-">
//   size   <width> <height>
//   K      <fx> <fy> <cx> <cy>
//   t      <tx> <ty> <tz>
//   R      <r00> <r01> <r02>
//           <r10> <r11> <r12>
//           <r20> <r21> <r22>
//   end
//
// `image` and `mask` are optional records identifying the corresponding files
// associated with the camera, used by the reconstruction pipeline.

struct CameraRecord {
    Camera      camera;
    std::string image_path;   // empty means none
    std::string mask_path;    // empty means none
};

bool save_camera_dataset(const std::string& path,
                         const std::vector<CameraRecord>& records);
bool load_camera_dataset(const std::string& path,
                         std::vector<CameraRecord>& out);

}  // namespace voxr
