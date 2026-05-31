// CUDA voxel ray-marcher. 1:1 port of render_cpu: one thread per pixel,
// 16x16 tiles, Amanatides-Woo 3D-DDA, Lambertian shade. Occupancy/color in
// global memory.
#include "voxr/render_cuda.hpp"

#include "cuda_common.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace voxr {
namespace {

// Grid params packed for the kernel.
struct DGrid {
    int   nx, ny, nz;
    float ox, oy, oz, vs;
};

__device__ inline std::size_t lin(const DGrid& g, int x, int y, int z) {
    return (std::size_t)x + (std::size_t)g.nx *
           ((std::size_t)y + (std::size_t)g.ny * (std::size_t)z);
}

// Ray vs grid AABB; entry/exit t. Mirrors ray_grid_aabb.
__device__ bool ray_aabb(const DGrid& g, float ro[3], float rd[3],
                        float& tmin, float& tmax) {
    float bmin[3] = {g.ox, g.oy, g.oz};
    float bmax[3] = {g.ox + g.nx * g.vs, g.oy + g.ny * g.vs, g.oz + g.nz * g.vs};
    float tlo = 0.f, thi = 1e30f;
    for (int a = 0; a < 3; ++a) {
        if (fabsf(rd[a]) < 1e-12f) {
            if (ro[a] < bmin[a] || ro[a] > bmax[a]) return false;
        } else {
            float inv = 1.f / rd[a];
            float t1 = (bmin[a] - ro[a]) * inv;
            float t2 = (bmax[a] - ro[a]) * inv;
            if (t1 > t2) { float t = t1; t1 = t2; t2 = t; }
            tlo = fmaxf(tlo, t1);
            thi = fminf(thi, t2);
            if (tlo > thi) return false;
        }
    }
    tmin = tlo; tmax = thi;
    return true;
}

__global__ void render_kernel(
        DGrid g, DCam cam, int W, int H,
        const std::uint8_t* occ, const std::uint8_t* cr,
        const std::uint8_t* cg, const std::uint8_t* cb,
        float bgr, float bgg, float bgb, int shading,
        float lx, float ly, float lz, float ambient, int max_steps,
        std::uint8_t* out) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    // unproject_ray: d_cam = ((u-cx)/fx,(v-cy)/fy,1); dir = R^T d_cam.
    float u = px + 0.5f, v = py + 0.5f;
    float dcx = (u - cam.cx) / cam.fx, dcy = (v - cam.cy) / cam.fy, dcz = 1.f;
    float dx = cam.R[0] * dcx + cam.R[3] * dcy + cam.R[6] * dcz;
    float dy = cam.R[1] * dcx + cam.R[4] * dcy + cam.R[7] * dcz;
    float dz = cam.R[2] * dcx + cam.R[5] * dcy + cam.R[8] * dcz;
    float dn = sqrtf(dx * dx + dy * dy + dz * dz);
    float rd[3] = {dx / dn, dy / dn, dz / dn};
    float ro[3] = {cam.px, cam.py, cam.pz};

    float col[3] = {bgr, bgg, bgb};
    float tmin, tmax;
    if (ray_aabb(g, ro, rd, tmin, tmax)) {
        const float eps = 1e-4f;
        float te = tmin + eps;
        float ex = ro[0] + rd[0] * te, ey = ro[1] + rd[1] * te,
              ez = ro[2] + rd[2] * te;
        float gx = (ex - g.ox) / g.vs, gy = (ey - g.oy) / g.vs,
              gz = (ez - g.oz) / g.vs;
        int ix = min(max((int)floorf(gx), 0), g.nx - 1);
        int iy = min(max((int)floorf(gy), 0), g.ny - 1);
        int iz = min(max((int)floorf(gz), 0), g.nz - 1);

        int sX = rd[0] > 0 ? 1 : (rd[0] < 0 ? -1 : 0);
        int sY = rd[1] > 0 ? 1 : (rd[1] < 0 ? -1 : 0);
        int sZ = rd[2] > 0 ? 1 : (rd[2] < 0 ? -1 : 0);
        const float inf = 1e30f;
        float tdX = sX ? g.vs / fabsf(rd[0]) : inf;
        float tdY = sY ? g.vs / fabsf(rd[1]) : inf;
        float tdZ = sZ ? g.vs / fabsf(rd[2]) : inf;
        auto fb = [&](int idx, int step, float roa, float rda, float oa) {
            if (step == 0) return inf;
            int b = step > 0 ? idx + 1 : idx;
            return (oa + b * g.vs - roa) / rda;
        };
        float tMX = fb(ix, sX, ro[0], rd[0], g.ox);
        float tMY = fb(iy, sY, ro[1], rd[1], g.oy);
        float tMZ = fb(iz, sZ, ro[2], rd[2], g.oz);

        int hit_axis = -1, hit_sign = 0;
        bool hit = false;
        for (int s = 0; s < max_steps; ++s) {
            if (ix < 0 || ix >= g.nx || iy < 0 || iy >= g.ny ||
                iz < 0 || iz >= g.nz) break;
            if (occ[lin(g, ix, iy, iz)]) { hit = true; break; }
            float tp;
            if (tMX < tMY && tMX < tMZ) {
                tp = tMX; ix += sX; tMX += tdX; hit_axis = 0; hit_sign = sX;
            } else if (tMY < tMZ) {
                tp = tMY; iy += sY; tMY += tdY; hit_axis = 1; hit_sign = sY;
            } else {
                tp = tMZ; iz += sZ; tMZ += tdZ; hit_axis = 2; hit_sign = sZ;
            }
            if (tp > tmax) break;
        }

        if (hit) {
            std::size_t idx = lin(g, ix, iy, iz);
            float base[3] = {cr[idx] / 255.f, cg[idx] / 255.f, cb[idx] / 255.f};
            if (shading && hit_axis >= 0) {
                float n[3] = {0.f, 0.f, 0.f};
                n[hit_axis] = (float)(-hit_sign);
                float ln = sqrtf(lx * lx + ly * ly + lz * lz);
                float lambert = fmaxf(0.f, (n[0] * lx + n[1] * ly + n[2] * lz) / ln);
                float k = ambient + (1.f - ambient) * lambert;
                col[0] = base[0] * k; col[1] = base[1] * k; col[2] = base[2] * k;
            } else {
                col[0] = base[0]; col[1] = base[1]; col[2] = base[2];
            }
        }
    }

    std::size_t o = ((std::size_t)py * W + px) * 3;
    out[o + 0] = (std::uint8_t)(fminf(fmaxf(col[0], 0.f), 1.f) * 255.f);
    out[o + 1] = (std::uint8_t)(fminf(fmaxf(col[1], 0.f), 1.f) * 255.f);
    out[o + 2] = (std::uint8_t)(fminf(fmaxf(col[2], 0.f), 1.f) * 255.f);
}

}  // namespace

// Device-resident state: the grid lives here across many render() calls.
struct CudaVoxelRenderer::Impl {
    DGrid g{};
    std::uint8_t *d_occ = nullptr, *d_cr = nullptr, *d_cg = nullptr,
                 *d_cb = nullptr, *d_out = nullptr;
    std::size_t out_cap = 0;  // bytes currently allocated for d_out
    cudaEvent_t e0{}, e1{}, e2{};
};

CudaVoxelRenderer::CudaVoxelRenderer(const VoxelGrid& grid) : impl_(new Impl) {
    const std::size_t N = grid.voxel_count();
    impl_->g = DGrid{grid.nx, grid.ny, grid.nz, grid.origin.x, grid.origin.y,
                     grid.origin.z, grid.voxel_size};
    VOXR_CUDA_CHECK(cudaEventCreate(&impl_->e0));
    VOXR_CUDA_CHECK(cudaEventCreate(&impl_->e1));
    VOXR_CUDA_CHECK(cudaEventCreate(&impl_->e2));

    VOXR_CUDA_CHECK(cudaMalloc(&impl_->d_occ, N));
    VOXR_CUDA_CHECK(cudaMalloc(&impl_->d_cr, N));
    VOXR_CUDA_CHECK(cudaMalloc(&impl_->d_cg, N));
    VOXR_CUDA_CHECK(cudaMalloc(&impl_->d_cb, N));

    // One-time grid upload, timed.
    VOXR_CUDA_CHECK(cudaEventRecord(impl_->e0));
    VOXR_CUDA_CHECK(cudaMemcpy(impl_->d_occ, grid.occupancy.data(), N, cudaMemcpyHostToDevice));
    VOXR_CUDA_CHECK(cudaMemcpy(impl_->d_cr, grid.color_r.data(), N, cudaMemcpyHostToDevice));
    VOXR_CUDA_CHECK(cudaMemcpy(impl_->d_cg, grid.color_g.data(), N, cudaMemcpyHostToDevice));
    VOXR_CUDA_CHECK(cudaMemcpy(impl_->d_cb, grid.color_b.data(), N, cudaMemcpyHostToDevice));
    VOXR_CUDA_CHECK(cudaEventRecord(impl_->e1));
    VOXR_CUDA_CHECK(cudaEventSynchronize(impl_->e1));
    VOXR_CUDA_CHECK(cudaEventElapsedTime(&upload_ms_, impl_->e0, impl_->e1));
}

CudaVoxelRenderer::~CudaVoxelRenderer() {
    if (!impl_) return;
    cudaFree(impl_->d_occ); cudaFree(impl_->d_cr);
    cudaFree(impl_->d_cg);  cudaFree(impl_->d_cb);
    if (impl_->d_out) cudaFree(impl_->d_out);
    cudaEventDestroy(impl_->e0);
    cudaEventDestroy(impl_->e1);
    cudaEventDestroy(impl_->e2);
    delete impl_;
}

bool CudaVoxelRenderer::render(const Camera& camera, ImageU8& out,
                               const RenderOptions& opts) {
    if (camera.width <= 0 || camera.height <= 0) {
        std::cerr << "CudaVoxelRenderer: invalid camera size\n";
        return false;
    }
    if (out.width != camera.width || out.height != camera.height ||
        out.channels != 3) {
        out = ImageU8(camera.width, camera.height, 3);
    }

    const int W = camera.width, H = camera.height;
    const std::size_t out_n = (std::size_t)W * H * 3;
    if (out_n > impl_->out_cap) {  // (re)allocate output buffer to fit
        if (impl_->d_out) cudaFree(impl_->d_out);
        VOXR_CUDA_CHECK(cudaMalloc(&impl_->d_out, out_n));
        impl_->out_cap = out_n;
    }

    DCam cam = to_dcam(camera);
    dim3 block(16, 16);
    dim3 gdim((W + 15) / 16, (H + 15) / 16);

    // Kernel only (grid already resident) -> the true per-frame compute cost.
    VOXR_CUDA_CHECK(cudaEventRecord(impl_->e0));
    render_kernel<<<gdim, block>>>(
        impl_->g, cam, W, H, impl_->d_occ, impl_->d_cr, impl_->d_cg, impl_->d_cb,
        opts.background.x, opts.background.y, opts.background.z,
        opts.shading ? 1 : 0, opts.light_direction.x, opts.light_direction.y,
        opts.light_direction.z, opts.ambient, opts.max_steps, impl_->d_out);
    VOXR_CUDA_CHECK(cudaGetLastError());
    VOXR_CUDA_CHECK(cudaEventRecord(impl_->e1));

    VOXR_CUDA_CHECK(cudaMemcpy(out.data.data(), impl_->d_out, out_n,
                               cudaMemcpyDeviceToHost));
    VOXR_CUDA_CHECK(cudaEventRecord(impl_->e2));
    VOXR_CUDA_CHECK(cudaEventSynchronize(impl_->e2));
    VOXR_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_, impl_->e0, impl_->e1));
    VOXR_CUDA_CHECK(cudaEventElapsedTime(&readback_ms_, impl_->e1, impl_->e2));
    return true;
}

bool render_cuda(const VoxelGrid& grid, const Camera& camera, ImageU8& out,
                 const RenderOptions& opts) {
    if (grid.voxel_count() == 0) {
        std::cerr << "render_cuda: empty grid\n";
        return false;
    }
    CudaVoxelRenderer r(grid);  // upload + single frame
    return r.render(camera, out, opts);
}

}  // namespace voxr
