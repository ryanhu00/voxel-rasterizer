#include "voxr/render_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

// =============================================================================
// GPU PARALLELIZATION STRATEGY (render_cuda.cu)
// =============================================================================
// Parallelism axis:
//   One CUDA thread per output pixel. Launch as a 2D grid over the image:
//       dim3 block(16, 16);                        // 256 threads / block
//       dim3 grid(ceil(W/16), ceil(H/16));
//   16x16 tiles match the typical screen-space coherence of DDA traversal —
//   neighbouring pixels follow nearly identical ray paths through the voxel
//   grid, so their texture/L1 fetches overlap heavily.
//
// Memory layout:
//   - camera params       -> __constant__ memory (one struct, read by every
//                           thread, perfect broadcast target).
//   - grid.occupancy      -> bind as a cudaTextureObject_t over a 3D cudaArray
//                           (or a surface object if we want writes too). The
//                           3D texture cache is what makes voxel DDA fast on
//                           GPUs — adjacent rays hit overlapping voxels and
//                           share cache lines. Falling back to a plain global
//                           uint8_t[] works but loses ~3-5x.
//   - grid.color_r/g/b    -> pack into a uchar4 3D texture (RGB + occupancy
//                           in .w) so a hit fetches color and shading data in
//                           one transaction.
//   - output image        -> global memory. Threads in a warp own consecutive
//                           x within a row -> 4-byte coalesced writes after
//                           we pack RGB into uchar4 on the device side and
//                           strip alpha when copying back to ImageU8.
//
// The big problem: warp divergence in DDA.
//   trace_ray() has a data-dependent loop count — some rays miss the AABB
//   entirely (exit immediately), some pierce the full diagonal (~3*N steps),
//   and rays in the same warp can differ by 100x in work. Mitigations, in
//   order of complexity:
//     1. Bound `max_steps` aggressively (already a parameter) and accept
//        that the warp waits for its slowest lane. Easiest, ~good enough.
//     2. Persistent-threads pattern: each thread pops pixels from a global
//        work queue when it finishes, so idle lanes get refilled. Recovers
//        most of the lost throughput at the cost of ~50 LOC of bookkeeping.
//     3. Wavefront/megakernel split: separate the "ray setup" kernel from
//        the "DDA step" kernel and compact active rays between launches.
//        Best throughput, only worth it for very deep grids (>1024^3).
//   Start with (1); profile; escalate only if the trace kernel is the
//   bottleneck (it usually is).
//
// Secondary win: early-ray-termination is implicit (return on first hit), so
// occupied scenes are *faster* than empty ones — opposite of most renderers.
//
// Expected speedup over the CPU striped-thread version:
//   ~30-100x at 1080p depending on scene density; trace divergence caps the
//   upper end relative to the reconstruction kernel.
// =============================================================================

namespace voxr {

namespace {

// Ray vs the grid's AABB; returns true on hit and stores entry/exit `t`.
// GPU: cheap and branch-light — keep as-is in the kernel. The 3-axis loop is
// fully unrolled by nvcc and the early-out on `tlo > thi` lets whole warps
// skip the expensive trace_ray() below when every ray misses the bbox (e.g.
// background pixels along the image border).
bool ray_grid_aabb(const VoxelGrid& g, Vec3 ro, Vec3 rd,
                   float& tmin, float& tmax) {
    Vec3 bmin = g.origin;
    Vec3 bmax = g.max_corner();
    float tlo = 0.f;
    float thi = 1e30f;
    for (int axis = 0; axis < 3; ++axis) {
        float o = ro[axis], d = rd[axis];
        float lo = bmin[axis], hi = bmax[axis];
        if (std::fabs(d) < 1e-12f) {
            if (o < lo || o > hi) return false;
        } else {
            float inv = 1.f / d;
            float t1 = (lo - o) * inv;
            float t2 = (hi - o) * inv;
            if (t1 > t2) std::swap(t1, t2);
            tlo = std::max(tlo, t1);
            thi = std::min(thi, t2);
            if (tlo > thi) return false;
        }
    }
    tmin = tlo;
    tmax = thi;
    return true;
}

// Encodes the result of a per-pixel trace. hit_axis identifies which slab
// boundary was crossed last (0/1/2 for X/Y/Z), used as a cheap face normal.
struct TraceResult {
    bool   hit{false};
    int    vx{-1}, vy{-1}, vz{-1};
    int    hit_axis{-1};
    int    hit_sign{0};   // +1 if ray stepped in + direction along hit_axis
    float  t_hit{0.f};
};

// Amanatides-Woo 3D-DDA traversal.
// GPU: this is THE hot kernel and the warp-divergence bottleneck.
//   - All scalars (ix/iy/iz, tMax*, tDelta*, step*) stay in registers — under
//     32 regs/thread, leaving plenty of occupancy headroom.
//   - The occupancy fetch on each step (`g.occupancy[linear_index(...)]`)
//     becomes `tex3D<uchar>(occ_tex, ix, iy, iz)`. The 3D texture cache turns
//     neighbouring threads' near-identical walks into shared cache lines.
//   - The `if (occupied) return` produces lane divergence — finished lanes
//     mask off while the rest of the warp keeps stepping. Acceptable for
//     modest `max_steps`; if profiling shows this dominates, switch to the
//     persistent-threads scheme described at the top of the file.
TraceResult trace_ray(const VoxelGrid& g, Vec3 ro, Vec3 rd, int max_steps) {
    TraceResult result;

    float tmin, tmax;
    if (!ray_grid_aabb(g, ro, rd, tmin, tmax)) return result;

    // Nudge slightly inside the grid so the entry voxel is unambiguous when
    // the ray strikes the AABB exactly on a face.
    const float eps   = 1e-4f;
    const float t_entry = tmin + eps;
    Vec3 entry = ro + rd * t_entry;
    Vec3 vp    = g.world_to_grid(entry);

    int ix = std::clamp(static_cast<int>(std::floor(vp.x)), 0, g.nx - 1);
    int iy = std::clamp(static_cast<int>(std::floor(vp.y)), 0, g.ny - 1);
    int iz = std::clamp(static_cast<int>(std::floor(vp.z)), 0, g.nz - 1);

    int stepX = rd.x > 0 ? 1 : (rd.x < 0 ? -1 : 0);
    int stepY = rd.y > 0 ? 1 : (rd.y < 0 ? -1 : 0);
    int stepZ = rd.z > 0 ? 1 : (rd.z < 0 ? -1 : 0);

    // tDelta_a = parametric distance to traverse exactly one voxel along axis a.
    const float inf = 1e30f;
    float tDeltaX = stepX ? g.voxel_size / std::fabs(rd.x) : inf;
    float tDeltaY = stepY ? g.voxel_size / std::fabs(rd.y) : inf;
    float tDeltaZ = stepZ ? g.voxel_size / std::fabs(rd.z) : inf;

    // tMax_a = absolute parametric value at which the ray crosses the NEXT
    // boundary along axis a, measured from `ro`.
    auto first_boundary = [&](int idx, int step, float ro_axis, float rd_axis,
                              float origin_axis) -> float {
        if (step == 0) return inf;
        int bnd_idx = step > 0 ? idx + 1 : idx;
        float bnd  = origin_axis + bnd_idx * g.voxel_size;
        return (bnd - ro_axis) / rd_axis;
    };

    float tMaxX = first_boundary(ix, stepX, ro.x, rd.x, g.origin.x);
    float tMaxY = first_boundary(iy, stepY, ro.y, rd.y, g.origin.y);
    float tMaxZ = first_boundary(iz, stepZ, ro.z, rd.z, g.origin.z);

    int hit_axis = -1;
    int hit_sign = 0;
    float t_prev = t_entry;

    // GPU: the step loop is the divergence hotspot. Lanes that hit early sit
    // idle while their warp-mates keep marching to `max_steps`. Keep the loop
    // bound tight and prefer texture fetches (below) over global loads so the
    // active lanes at least run at peak bandwidth.
    for (int s = 0; s < max_steps; ++s) {
        if (!g.in_bounds(ix, iy, iz)) break;

        // GPU: replace with `tex3D<uint8_t>(occ_tex, ix, iy, iz)` — same
        // semantics, but reads go through the 3D texture cache that warps
        // share for spatially coherent rays.
        if (g.occupancy[g.linear_index(ix, iy, iz)]) {
            result.hit      = true;
            result.vx       = ix;
            result.vy       = iy;
            result.vz       = iz;
            result.hit_axis = hit_axis;
            result.hit_sign = hit_sign;
            result.t_hit    = t_prev;
            return result;
        }

        // Advance to the closest of the three boundaries.
        if (tMaxX < tMaxY && tMaxX < tMaxZ) {
            t_prev    = tMaxX;
            ix       += stepX;
            tMaxX    += tDeltaX;
            hit_axis  = 0;
            hit_sign  = stepX;
        } else if (tMaxY < tMaxZ) {
            t_prev    = tMaxY;
            iy       += stepY;
            tMaxY    += tDeltaY;
            hit_axis  = 1;
            hit_sign  = stepY;
        } else {
            t_prev    = tMaxZ;
            iz       += stepZ;
            tMaxZ    += tDeltaZ;
            hit_axis  = 2;
            hit_sign  = stepZ;
        }

        if (t_prev > tmax) break;
    }
    return result;
}

// GPU: shading is dirt-cheap relative to tracing — inline it directly into
// the trace kernel rather than launching a second pass. The branch on
// `tr.hit` is uniform across most warps (whole tiles tend to all hit or all
// miss against compact objects), so divergence here is negligible.
void shade_pixel(const VoxelGrid& g, const RenderOptions& opts,
                 const TraceResult& tr, Vec3& out_rgb_01) {
    if (!tr.hit) {
        out_rgb_01 = opts.background;
        return;
    }
    std::size_t idx = g.linear_index(tr.vx, tr.vy, tr.vz);
    Vec3 base{
        g.color_r[idx] / 255.f,
        g.color_g[idx] / 255.f,
        g.color_b[idx] / 255.f
    };

    if (!opts.shading || tr.hit_axis < 0) {
        out_rgb_01 = base;
        return;
    }
    // Face normal points opposite the step direction along the hit axis.
    Vec3 n{0.f, 0.f, 0.f};
    (&n.x)[tr.hit_axis] = static_cast<float>(-tr.hit_sign);

    Vec3 L = normalize(opts.light_direction);
    float lambert = std::max(0.f, dot(n, L));
    float k = opts.ambient + (1.f - opts.ambient) * lambert;
    out_rgb_01 = base * k;
}

}  // namespace

bool render_cpu(const VoxelGrid& grid, const Camera& camera, ImageU8& out,
                const RenderOptions& opts) {
    if (camera.width <= 0 || camera.height <= 0) {
        std::cerr << "render_cpu: invalid camera size\n";
        return false;
    }
    if (grid.voxel_count() == 0) {
        std::cerr << "render_cpu: empty grid\n";
        return false;
    }
    if (out.width != camera.width || out.height != camera.height ||
        out.channels != 3) {
        out = ImageU8(camera.width, camera.height, 3);
    }

    // GPU: this whole row-stealing scheduler collapses to a single
    // `render_kernel<<<grid, block>>>` launch. The block grid (W/16, H/16)
    // implicitly walks the image in 16x16 tiles, which is better for ray
    // coherence than the CPU's row-at-a-time scheme — adjacent rays within a
    // tile share voxel-cache lines, where the CPU's row workers don't share
    // anything across thread boundaries.
    const unsigned n_workers =
        std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next_row{0};
    std::vector<std::thread> workers;
    workers.reserve(n_workers);

    for (unsigned w = 0; w < n_workers; ++w) {
        workers.emplace_back([&]() {
            while (true) {
                int y = next_row.fetch_add(1, std::memory_order_relaxed);
                if (y >= camera.height) break;
                // GPU: the (x, y) loops vanish — x = blockIdx.x*16+threadIdx.x,
                // y = blockIdx.y*16+threadIdx.y. The body below is the kernel.
                for (int x = 0; x < camera.width; ++x) {
                    Vec3 ro, rd;
                    camera.unproject_ray(static_cast<float>(x) + 0.5f,
                                         static_cast<float>(y) + 0.5f,
                                         ro, rd);
                    TraceResult tr = trace_ray(grid, ro, rd, opts.max_steps);
                    Vec3 col;
                    shade_pixel(grid, opts, tr, col);
                    // GPU: pack into a uchar4 surface write so the warp emits
                    // one 64-byte coalesced transaction per 16 pixels instead
                    // of 3 strided byte writes. Strip alpha during the
                    // device->host copy back into the ImageU8 RGB buffer.
                    std::size_t idx = out.pixel_index(x, y);
                    out.data[idx + 0] = static_cast<std::uint8_t>(
                        clampf(col.x, 0.f, 1.f) * 255.f);
                    out.data[idx + 1] = static_cast<std::uint8_t>(
                        clampf(col.y, 0.f, 1.f) * 255.f);
                    out.data[idx + 2] = static_cast<std::uint8_t>(
                        clampf(col.z, 0.f, 1.f) * 255.f);
                }
            }
        });
    }
    for (auto& t : workers) t.join();
    return true;
}

}  // namespace voxr
