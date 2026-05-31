// Shared device-side POD + helpers for the CUDA kernels.
#pragma once

#include "voxr/camera.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define VOXR_CUDA_CHECK(call)                                              \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,    \
                         __FILE__, __LINE__, cudaGetErrorString(err__));   \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

namespace voxr {

// Flattened, register-friendly camera (mirrors voxr::Camera). R is row-major.
struct DCam {
    int   w, h;
    float fx, fy, cx, cy;
    float px, py, pz;
    float R[9];
};

inline DCam to_dcam(const Camera& c) {
    DCam d;
    d.w = c.width;  d.h = c.height;
    d.fx = c.fx; d.fy = c.fy; d.cx = c.cx; d.cy = c.cy;
    d.px = c.position.x; d.py = c.position.y; d.pz = c.position.z;
    for (int i = 0; i < 9; ++i) d.R[i] = c.R.m[i];
    return d;
}

// x_cam = R * (p - position); project. False if behind the camera.
__device__ inline bool dcam_project(const DCam& c, float wx, float wy, float wz,
                                    float& u, float& v) {
    float dx = wx - c.px, dy = wy - c.py, dz = wz - c.pz;
    float zc = c.R[6] * dx + c.R[7] * dy + c.R[8] * dz;
    if (zc <= 1e-6f) return false;
    float xc = c.R[0] * dx + c.R[1] * dy + c.R[2] * dz;
    float yc = c.R[3] * dx + c.R[4] * dy + c.R[5] * dz;
    u = c.fx * xc / zc + c.cx;
    v = c.fy * yc / zc + c.cy;
    return true;
}

}  // namespace voxr
