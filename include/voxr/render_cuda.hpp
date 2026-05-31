// CUDA voxel-grid renderer. Drop-in for render_cpu: one thread per pixel,
// Amanatides-Woo 3D-DDA, occupancy/color in global memory.
#pragma once

#include "voxr/render_cpu.hpp"  // RenderOptions, types

namespace voxr {

// Grid-resident renderer: uploads the voxel grid to the device ONCE, then
// renders any number of frames with only a kernel launch + readback per call.
// This is the "live GPU loop" path — used for orbits and the interactive
// viewer, where re-uploading the grid every frame (as the one-shot
// render_cuda below does) would dominate the cost.
//
// pImpl keeps CUDA types out of this header so plain C++ apps can include it.
class CudaVoxelRenderer {
public:
    explicit CudaVoxelRenderer(const VoxelGrid& grid);
    ~CudaVoxelRenderer();
    CudaVoxelRenderer(const CudaVoxelRenderer&)            = delete;
    CudaVoxelRenderer& operator=(const CudaVoxelRenderer&) = delete;

    // Renders `camera` into `out` (resized to camera size if needed).
    bool render(const Camera& camera, ImageU8& out,
                const RenderOptions& opts = {});

    // Milliseconds, measured with CUDA events.
    float upload_ms()        const { return upload_ms_; }    // one-time
    float last_kernel_ms()   const { return kernel_ms_; }    // last render()
    float last_readback_ms() const { return readback_ms_; }  // last render()

private:
    struct Impl;
    Impl* impl_;
    float upload_ms_{0.f}, kernel_ms_{0.f}, readback_ms_{0.f};
};

// One-shot convenience: uploads the grid, renders one frame, frees. Same
// contract as render_cpu. Prefer CudaVoxelRenderer for multi-frame loops.
bool render_cuda(const VoxelGrid&     grid,
                 const Camera&        camera,
                 ImageU8&             out,
                 const RenderOptions& opts = {});

}  // namespace voxr
