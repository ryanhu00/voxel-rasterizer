#include "voxr/reconstruct_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

// =============================================================================
// GPU PARALLELIZATION STRATEGY (reconstruct_cuda.cu)
// =============================================================================
// This file is the CPU reference; the structure below is intentionally written
// so each "kernel body" maps 1:1 to a CUDA thread.
//
// Parallelism axis:
//   Every voxel (x, y, z) is independent — no inter-voxel dependencies, no
//   atomics, no reductions. This is the ideal embarrassingly-parallel case.
//   Launch one thread per voxel:
//       dim3 block(8, 8, 8);                       // 512 threads / block
//       dim3 grid(ceil(nx/8), ceil(ny/8), ceil(nz/8));
//   8^3 is a good starting point because it matches typical voxel locality and
//   keeps the block under the 1024-thread cap with room for register pressure.
//
// Memory layout:
//   - cameras[]            -> __constant__ memory (small, broadcast to all
//                            threads in a warp; same camera params reread per
//                            voxel per camera). Cap ~64 KB; if cam count is
//                            larger, fall back to global + L1.
//   - masks[], images[]    -> cudaTextureObject_t per view. Textures give us
//                            (a) free hardware bilinear filtering — replacing
//                            sample_mask_bilinear / sample_rgb_bilinear with
//                            tex2D() calls, (b) clamp-to-border addressing so
//                            the out-of-bounds branch disappears, and (c) 2D
//                            spatial cache that neighbouring threads share
//                            when they project to nearby pixels.
//   - grid.occupancy/color -> global memory, one write per thread. Writes are
//                            naturally coalesced if grid.linear_index uses the
//                            standard x-fastest layout (threadIdx.x neighbours
//                            write contiguous bytes).
//
// Divergence and load balance:
//   The inner loop over cameras is uniform across a warp (all threads iterate
//   the same num_cams), so the only divergence is the cam.project() / mask
//   threshold branch. That's mild — warps spend roughly the same work on
//   "inside silhouette" vs "outside" voxels.
//
// Expected speedup over the CPU striped-thread version:
//   ~50-200x for a 256^3 grid with 16 views, dominated by texture-sampled
//   bilinear fetches and the lack of any serialization.
// =============================================================================

namespace voxr {

namespace {

// Bilinear-sampled mask value in [0, 255] at floating pixel (u, v). Returns
// 0 (background) when the projection lies outside the image.
// GPU: replace this entire function with `tex2D<float>(mask_tex, u, v)`. A
// cudaTextureObject_t configured with cudaFilterModeLinear and
// cudaAddressModeBorder gives identical semantics in a single instruction.
inline float sample_mask_bilinear(const ImageU8& mask, float u, float v) {
    if (u < 0.f || v < 0.f) return 0.f;
    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= mask.width || y1 >= mask.height) return 0.f;
    float fx = u - x0;
    float fy = v - y0;
    auto at = [&](int x, int y) -> float {
        return static_cast<float>(mask.data[mask.pixel_index(x, y)]);
    };
    float a = at(x0, y0) * (1.f - fx) + at(x1, y0) * fx;
    float b = at(x0, y1) * (1.f - fx) + at(x1, y1) * fx;
    return a * (1.f - fy) + b * fy;
}

// Bilinear-sampled RGB at floating pixel (u, v) on an interleaved RGB image.
// GPU: use a uchar4 texture (pad RGB->RGBA on upload) and `tex2D<float4>` for
// a single coalesced 4-byte fetch with hardware bilinear. uchar4 is preferred
// over float4 here — 4x smaller texture footprint, same filtering quality.
inline bool sample_rgb_bilinear(const ImageU8& img, float u, float v,
                                Vec3& out) {
    if (img.channels != 3) return false;
    if (u < 0.f || v < 0.f) return false;
    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= img.width || y1 >= img.height) return false;
    float fx = u - x0;
    float fy = v - y0;
    auto at = [&](int x, int y, int c) -> float {
        return static_cast<float>(img.data[img.pixel_index(x, y) + c]);
    };
    for (int c = 0; c < 3; ++c) {
        float a = at(x0, y0, c) * (1.f - fx) + at(x1, y0, c) * fx;
        float b = at(x0, y1, c) * (1.f - fx) + at(x1, y1, c) * fx;
        (&out.x)[c] = (a * (1.f - fy) + b * fy);
    }
    return true;
}

// Process one z-slice of the grid. This is the "kernel body" of the CPU
// prototype: it does exactly what one CUDA block-row of threads will do later.
// GPU: the (x, y) double loop disappears — those become threadIdx.x/y across
// the launch grid, and `z` becomes threadIdx.z + blockIdx.z * blockDim.z. The
// function body from the inner loop onward becomes the kernel verbatim.
void process_slice(int z,
                   VoxelGrid& grid,
                   const std::vector<Camera>& cameras,
                   const std::vector<ImageU8>& masks,
                   const std::vector<ImageU8>& images,
                   const ReconstructOptions& opts,
                   int required) {
    const int nx = grid.nx, ny = grid.ny;
    const int num_cams = static_cast<int>(cameras.size());
    const bool fuse = opts.fuse_color && !images.empty();

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            Vec3 p_world = grid.voxel_center(x, y, z);

            int   consistent = 0;
            Vec3  color_sum{0.f, 0.f, 0.f};
            int   color_count = 0;

            // GPU: this camera loop is uniform across a warp — every thread
            // visits the same camera index in lockstep. That lets us preload
            // `cam` into a register once per iteration (it's already in
            // __constant__ memory, so the broadcast is free) and incurs no
            // divergence until the project()/mask-threshold branch below.
            for (int ci = 0; ci < num_cams; ++ci) {
                const Camera& cam = cameras[ci];
                float u, v, depth;
                if (!cam.project(p_world, u, v, depth)) continue;
                if (u < 0.f || v < 0.f || u >= cam.width || v >= cam.height)
                    continue;
                float m = sample_mask_bilinear(masks[ci], u, v);
                if (m >= static_cast<float>(opts.silhouette_threshold)) {
                    consistent += 1;
                    if (fuse && ci < static_cast<int>(images.size())) {
                        Vec3 col;
                        if (sample_rgb_bilinear(images[ci], u, v, col)) {
                            color_sum += col;
                            color_count += 1;
                        }
                    }
                }
            }

            // GPU: each thread owns a unique `idx`, so the write below needs
            // no atomics. With x-fastest linear indexing, threads in a warp
            // (consecutive threadIdx.x) write consecutive bytes — perfectly
            // coalesced for both `occupancy` and the three color planes.
            const std::size_t idx = grid.linear_index(x, y, z);
            if (consistent >= required) {
                grid.occupancy[idx] = 1;
                if (fuse && color_count > 0) {
                    Vec3 avg = color_sum / static_cast<float>(color_count);
                    grid.color_r[idx] = static_cast<std::uint8_t>(
                        clampf(avg.x, 0.f, 255.f));
                    grid.color_g[idx] = static_cast<std::uint8_t>(
                        clampf(avg.y, 0.f, 255.f));
                    grid.color_b[idx] = static_cast<std::uint8_t>(
                        clampf(avg.z, 0.f, 255.f));
                } else {
                    grid.color_r[idx] = 200;
                    grid.color_g[idx] = 200;
                    grid.color_b[idx] = 200;
                }
            } else {
                grid.occupancy[idx] = 0;
            }
        }
    }
}

}  // namespace

void reconstruct_cpu(VoxelGrid& grid,
                     const std::vector<Camera>& cameras,
                     const std::vector<ImageU8>& masks,
                     const std::vector<ImageU8>& images,
                     const ReconstructOptions& opts) {
    const int num_cams = static_cast<int>(cameras.size());
    int required = opts.min_consistent_views;
    if (required < 0 || required > num_cams) required = num_cams;

    // Crude but effective: launch one worker per hardware thread, partition
    // slices in a striped fashion to balance load (a sphere's middle slices
    // touch more cameras than the polar caps).
    // GPU: this entire scheduler is replaced by a single kernel launch. The
    // hardware warp scheduler handles the "striped load balance" implicitly —
    // any block that finishes its 8^3 tile early gets retired and the SM
    // picks up the next one. Host-side work shrinks to:
    //     upload cameras (constant mem) + masks/images (textures),
    //     launch reconstruct_kernel<<<grid, block>>>(...),
    //     download grid.occupancy + color planes.
    const unsigned n_workers =
        std::max(1u, std::thread::hardware_concurrency());

    std::atomic<int> next_z{0};
    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for (unsigned w = 0; w < n_workers; ++w) {
        workers.emplace_back([&]() {
            while (true) {
                int z = next_z.fetch_add(1, std::memory_order_relaxed);
                if (z >= grid.nz) break;
                process_slice(z, grid, cameras, masks, images, opts, required);
            }
        });
    }
    for (auto& t : workers) t.join();
}

}  // namespace voxr
