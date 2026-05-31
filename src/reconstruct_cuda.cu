// CUDA shape-from-silhouette. 1:1 port of reconstruct_cpu's per-voxel body:
// one thread per voxel, 8x8x8 blocks, masks/images in global memory.
#include "voxr/reconstruct_cuda.hpp"

#include "cuda_common.cuh"

#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

namespace voxr {
namespace {

// Bilinear mask value in [0,255]; 0 (background) when the projection is out of
// bounds. Mirrors sample_mask_bilinear on the CPU.
__device__ float sample_mask(const std::uint8_t* m, int w, int h,
                             float u, float v) {
    if (u < 0.f || v < 0.f) return 0.f;
    int x0 = (int)floorf(u), y0 = (int)floorf(v);
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x1 >= w || y1 >= h) return 0.f;
    float fx = u - x0, fy = v - y0;
    float a = m[y0 * w + x0] * (1.f - fx) + m[y0 * w + x1] * fx;
    float b = m[y1 * w + x0] * (1.f - fx) + m[y1 * w + x1] * fx;
    return a * (1.f - fy) + b * fy;
}

// Bilinear RGB on an interleaved image. False when out of bounds.
__device__ bool sample_rgb(const std::uint8_t* img, int w, int h,
                          float u, float v, float& r, float& g, float& b) {
    if (u < 0.f || v < 0.f) return false;
    int x0 = (int)floorf(u), y0 = (int)floorf(v);
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x1 >= w || y1 >= h) return false;
    float fx = u - x0, fy = v - y0;
    auto px = [&](int x, int y, int c) { return (float)img[(y * w + x) * 3 + c]; };
    float out[3];
    for (int c = 0; c < 3; ++c) {
        float aa = px(x0, y0, c) * (1.f - fx) + px(x1, y0, c) * fx;
        float bb = px(x0, y1, c) * (1.f - fx) + px(x1, y1, c) * fx;
        out[c] = aa * (1.f - fy) + bb * fy;
    }
    r = out[0]; g = out[1]; b = out[2];
    return true;
}

__global__ void reconstruct_kernel(
        int nx, int ny, int nz, float ox, float oy, float oz, float vs,
        const DCam* cams, int num_cams,
        const std::uint8_t* const* masks,
        const std::uint8_t* const* images,  // null when not fusing color
        int required, int threshold,
        std::uint8_t* occ, std::uint8_t* cr, std::uint8_t* cg,
        std::uint8_t* cb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    float wx = ox + (x + 0.5f) * vs;
    float wy = oy + (y + 0.5f) * vs;
    float wz = oz + (z + 0.5f) * vs;

    int   consistent = 0, color_count = 0;
    float sr = 0.f, sg = 0.f, sb = 0.f;

    for (int ci = 0; ci < num_cams; ++ci) {
        const DCam& cam = cams[ci];
        float u, v;
        if (!dcam_project(cam, wx, wy, wz, u, v)) continue;
        if (u < 0.f || v < 0.f || u >= cam.w || v >= cam.h) continue;
        if (sample_mask(masks[ci], cam.w, cam.h, u, v) >= (float)threshold) {
            ++consistent;
            if (images) {
                float r, g, b;
                if (sample_rgb(images[ci], cam.w, cam.h, u, v, r, g, b)) {
                    sr += r; sg += g; sb += b; ++color_count;
                }
            }
        }
    }

    std::size_t idx = (std::size_t)x + (std::size_t)nx *
                      ((std::size_t)y + (std::size_t)ny * (std::size_t)z);
    if (consistent >= required) {
        occ[idx] = 1;
        if (images && color_count > 0) {
            float inv = 1.f / color_count;
            cr[idx] = (std::uint8_t)fminf(255.f, fmaxf(0.f, sr * inv));
            cg[idx] = (std::uint8_t)fminf(255.f, fmaxf(0.f, sg * inv));
            cb[idx] = (std::uint8_t)fminf(255.f, fmaxf(0.f, sb * inv));
        } else {
            cr[idx] = 200; cg[idx] = 200; cb[idx] = 200;
        }
    } else {
        occ[idx] = 0;
    }
}

}  // namespace

void reconstruct_cuda(VoxelGrid& grid,
                      const std::vector<Camera>& cameras,
                      const std::vector<ImageU8>& masks,
                      const std::vector<ImageU8>& images,
                      const ReconstructOptions& opts) {
    const int num_cams = (int)cameras.size();
    int required = opts.min_consistent_views;
    if (required < 0 || required > num_cams) required = num_cams;
    const bool fuse = opts.fuse_color && !images.empty();
    const std::size_t N = grid.voxel_count();

    // Cameras.
    std::vector<DCam> h_cams(num_cams);
    for (int i = 0; i < num_cams; ++i) h_cams[i] = to_dcam(cameras[i]);
    DCam* d_cams = nullptr;
    VOXR_CUDA_CHECK(cudaMalloc(&d_cams, num_cams * sizeof(DCam)));
    VOXR_CUDA_CHECK(cudaMemcpy(d_cams, h_cams.data(),
                               num_cams * sizeof(DCam), cudaMemcpyHostToDevice));

    // Per-view mask (and optionally image) buffers + array of device pointers.
    std::vector<std::uint8_t*> h_mask(num_cams, nullptr);
    std::vector<std::uint8_t*> h_img(num_cams, nullptr);
    for (int i = 0; i < num_cams; ++i) {
        std::size_t mn = masks[i].data.size();
        VOXR_CUDA_CHECK(cudaMalloc(&h_mask[i], mn));
        VOXR_CUDA_CHECK(cudaMemcpy(h_mask[i], masks[i].data.data(), mn,
                                   cudaMemcpyHostToDevice));
        if (fuse && i < (int)images.size() && images[i].channels == 3) {
            std::size_t in = images[i].data.size();
            VOXR_CUDA_CHECK(cudaMalloc(&h_img[i], in));
            VOXR_CUDA_CHECK(cudaMemcpy(h_img[i], images[i].data.data(), in,
                                       cudaMemcpyHostToDevice));
        }
    }
    std::uint8_t** d_maskPtrs = nullptr;
    std::uint8_t** d_imgPtrs = nullptr;
    VOXR_CUDA_CHECK(cudaMalloc(&d_maskPtrs, num_cams * sizeof(std::uint8_t*)));
    VOXR_CUDA_CHECK(cudaMemcpy(d_maskPtrs, h_mask.data(),
                               num_cams * sizeof(std::uint8_t*),
                               cudaMemcpyHostToDevice));
    if (fuse) {
        VOXR_CUDA_CHECK(cudaMalloc(&d_imgPtrs, num_cams * sizeof(std::uint8_t*)));
        VOXR_CUDA_CHECK(cudaMemcpy(d_imgPtrs, h_img.data(),
                                   num_cams * sizeof(std::uint8_t*),
                                   cudaMemcpyHostToDevice));
    }

    // Output channels.
    std::uint8_t *d_occ, *d_cr, *d_cg, *d_cb;
    VOXR_CUDA_CHECK(cudaMalloc(&d_occ, N));
    VOXR_CUDA_CHECK(cudaMalloc(&d_cr, N));
    VOXR_CUDA_CHECK(cudaMalloc(&d_cg, N));
    VOXR_CUDA_CHECK(cudaMalloc(&d_cb, N));

    dim3 block(8, 8, 8);
    dim3 gdim((grid.nx + 7) / 8, (grid.ny + 7) / 8, (grid.nz + 7) / 8);

    // Optional kernel-only timing (set VOXR_CUDA_TIMING=1).
    const bool timing = std::getenv("VOXR_CUDA_TIMING") != nullptr;
    cudaEvent_t e0, e1;
    if (timing) {
        VOXR_CUDA_CHECK(cudaEventCreate(&e0));
        VOXR_CUDA_CHECK(cudaEventCreate(&e1));
        VOXR_CUDA_CHECK(cudaEventRecord(e0));
    }

    reconstruct_kernel<<<gdim, block>>>(
        grid.nx, grid.ny, grid.nz, grid.origin.x, grid.origin.y, grid.origin.z,
        grid.voxel_size, d_cams, num_cams, d_maskPtrs, fuse ? d_imgPtrs : nullptr,
        required, opts.silhouette_threshold, d_occ, d_cr, d_cg, d_cb);
    VOXR_CUDA_CHECK(cudaGetLastError());
    VOXR_CUDA_CHECK(cudaDeviceSynchronize());

    if (timing) {
        float ms = 0.f;
        VOXR_CUDA_CHECK(cudaEventRecord(e1));
        VOXR_CUDA_CHECK(cudaEventSynchronize(e1));
        VOXR_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::fprintf(stderr, "[voxr] reconstruct_kernel: %.3f ms (%d^3 voxels, "
                     "%d views)\n", ms, grid.nx, num_cams);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }

    VOXR_CUDA_CHECK(cudaMemcpy(grid.occupancy.data(), d_occ, N, cudaMemcpyDeviceToHost));
    VOXR_CUDA_CHECK(cudaMemcpy(grid.color_r.data(), d_cr, N, cudaMemcpyDeviceToHost));
    VOXR_CUDA_CHECK(cudaMemcpy(grid.color_g.data(), d_cg, N, cudaMemcpyDeviceToHost));
    VOXR_CUDA_CHECK(cudaMemcpy(grid.color_b.data(), d_cb, N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_cams; ++i) {
        cudaFree(h_mask[i]);
        if (h_img[i]) cudaFree(h_img[i]);
    }
    cudaFree(d_maskPtrs);
    if (d_imgPtrs) cudaFree(d_imgPtrs);
    cudaFree(d_cams);
    cudaFree(d_occ); cudaFree(d_cr); cudaFree(d_cg); cudaFree(d_cb);
}

}  // namespace voxr
