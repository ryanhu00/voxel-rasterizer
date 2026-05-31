# voxel-rasterizer

GPU-accelerated multi-view voxel reconstruction + rasterization
(CS179, Ryan Hu & Samuel Xie). Given calibrated images, reconstruct a 3D
occupancy + color grid via shape-from-silhouette, then ray-march novel views.

Both kernel-like loops (per-voxel SfS, per-pixel ray-march) now have **CUDA
ports** alongside the CPU reference. The CPU paths stay as the validation
oracle — `test_cuda_parity` pins the GPU output to them. Pass `--gpu` to
`reconstruct`/`render` to use the device.

## Layout

```
include/voxr/    Public headers
src/             CPU library
apps/            CLI tools
scripts/         Pipeline drivers
tests/           ctest binaries
data/            Generated datasets (git-ignored)
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is detected automatically: if `nvcc` is on the toolchain, the `voxr_cuda`
library and the `--gpu` flag light up; otherwise the build is CPU-only and
identical. GPU arch defaults to `native` (override with
`-DCMAKE_CUDA_ARCHITECTURES=86`).

| Binary | Purpose |
| --- | --- |
| `synth_dataset` | Render an analytic shape (sphere/cube/dumbbell) from a ring of cameras → RGB + masks + `cameras.txt`. |
| `reconstruct` | Shape-from-silhouette → voxel grid. `--gpu` for the CUDA kernel. |
| `render` | Render a voxel grid to PPM (single view or orbit). `--gpu` for the CUDA ray-marcher. |
| `bake_views` | Pre-render an azimuth × elevation × radius grid + self-contained HTML viewer (drag = orbit, scroll = zoom). `--gpu` renders the frames with the resident CUDA loop. |
| `make_orbit_cameras` | Write `cameras.txt` for a folder of ring-orbit photos. |

## Demo (sphere)

```bash
build/synth_dataset --out data/sphere --shape sphere --views 24 --res 256 256
build/reconstruct  --in data/sphere --out data/sphere/voxels.bin --grid 128
build/render       --voxels data/sphere/voxels.bin --out data/sphere/novel.ppm \
                   --eye 1.6 1.0 1.6 --target 0 0 0 --up 0 1 0 --res 512 512

# Orbit + mp4 (needs netpbm + ffmpeg)
build/render --voxels data/sphere/voxels.bin --orbit data/sphere/orbit \
             --views 36 --radius 3 --res 512 512
for f in data/sphere/orbit/*.ppm; do pnmtopng "$f" > "${f%.ppm}.png"; done
ffmpeg -y -framerate 24 -i data/sphere/orbit/frame_%04d.png \
       -c:v libx264 -pix_fmt yuv420p data/sphere/orbit.mp4

# Interactive viewer
build/bake_views --voxels data/sphere/voxels.bin --out data/sphere/view
cd data/sphere/view && python3 -m http.server 8000
# open http://localhost:8000/viewer.html
```

Swap `--shape cube` or `--shape dumbbell` to test the visual hull on concavity.

## GPU

Append `--gpu` to `reconstruct` or `render` to run the CUDA kernels instead of
the CPU threads — same flags, same output. Layout matches the strategy notes
in [src/reconstruct_cpu.cpp](src/reconstruct_cpu.cpp) and
[src/render_cpu.cpp](src/render_cpu.cpp): reconstruction is one thread per
voxel (`8x8x8` blocks), rendering is one thread per pixel (`16x16` tiles).

```bash
build/reconstruct --in data/sphere --out data/sphere/voxels.bin --grid 256 --gpu
build/render      --voxels data/sphere/voxels.bin --out novel.ppm \
                  --eye 1.6 1.0 1.6 --res 1024 1024 --gpu
```

For multi-frame work the GPU renderer is **grid-resident**: orbits and the
interactive viewer upload the voxel grid to the device once, then every frame
is just a kernel launch + readback.

```bash
build/render --voxels data/sphere/voxels.bin --orbit out --views 36 --gpu  # resident loop
build/render --voxels data/sphere/voxels.bin --bench 200                   # isolated GPU timings
build/bake_views --voxels data/sphere/voxels.bin --out view --gpu          # GPU-rendered web viewer
```

Sphere (24 views, RTX A5000, vs 16-thread CPU):

| Stage | CPU | GPU | Note |
| --- | --- | --- | --- |
| Reconstruct 256³ | 2.01 s | 0.52 s wall · **13.9 ms** kernel | end-to-end wall shares disk I/O + CUDA init |
| Render 1024² | — | **1.70 ms** kernel (≈480 fps) | one-time 9 ms grid upload |
| Orbit 36×512² | 1.85 s | 0.57 s | resident loop, grid uploaded once |

Full breakdown + `ncu` kernel profiles (occupancy, throughput, warp stalls) and
what the profile says to optimize next: **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)**.
Short version: the renderer is compute-bound at ~full occupancy; reconstruction
is bound by the per-voxel mask gather (82 % memory pipe), which is exactly what
texture memory would fix in week 3.

## Your own photos

[`scripts/photos_to_voxels.sh`](scripts/photos_to_voxels.sh) drives the full
pipeline. Auto-converts PNG/JPG/PPM, masks via `rembg`, writes cameras,
reconstructs, bakes the viewer.

```bash
scripts/photos_to_voxels.sh ~/photos/bottle data/bottle \
    -- --radius 5.0 --fov 1.2 --elev 0

cd data/bottle/view && python3 -m http.server 8000
```

Top-level flags: `--masks-dir DIR` (skip rembg), `--no-mask` (stop after
cameras.txt), `--grid N`. Flags after `--` go to `make_orbit_cameras`
(`--radius`, `--elev`, `--fov`, `--target`, `--start-deg`, `--cw`).

Assumes a **ring orbit** capture (constant radius + elevation), alphabetical
filename order = shoot order. Freehand capture needs COLMAP + a hand-written
`cameras.txt` — format in [include/voxr/camera.hpp](include/voxr/camera.hpp).
Sparse reconstruction usually means the orbit flags don't match the real
shoot; try `--min-views N` lower than the photo count to debug.

Tools: `pngtopnm`/`jpegtopnm` (netpbm) or `convert` (ImageMagick);
`pip install rembg` for auto-masking.

## File formats

- **PPM** (`P6`, 8-bit RGB) for color, **PGM** (`P5`, 8-bit; ≥128 = foreground)
  for masks.
- **`cameras.txt`** — intrinsics + `t` + `R` per camera. Schema in
  [include/voxr/camera.hpp](include/voxr/camera.hpp).
- **`*.bin`** — `VOXG` magic, version 1, little-endian: dims, origin,
  voxel_size, then 4 uint8 channels (occupancy, r, g, b) in
  `idx = x + nx*(y + ny*z)` order. Layout in
  [include/voxr/voxel_grid.hpp](include/voxr/voxel_grid.hpp).

## Conventions

OpenCV-style camera: +X right, +Y down, +Z forward, right-handed.
World right-handed, +Y up by default. Projection is
`pixel = K · R · (X_world − t_world)` divided by `z_cam`.

## Tests

Header-only harness in [tests/test_harness.hpp](tests/test_harness.hpp)
(`VOXR_EXPECT` / `VOXR_EXPECT_NEAR`; failure count = exit code).

| Test | What it pins down |
| --- | --- |
| [`test_camera`](tests/test_camera.cpp) | Project ↔ unproject round-trip; principal-point and axis-direction sanity; behind-camera rejection. |
| [`test_voxel_grid`](tests/test_voxel_grid.cpp) | `x + nx*(y + ny*z)` indexing; `voxel_center` / `world_to_grid` inverse; full `VOXG` save/load round-trip. |
| [`test_pipeline`](tests/test_pipeline.cpp) | Synth → reconstruct → render. Requires **recall > 0.95** and **bloat < 0.40**. The CUDA validation oracle. |
| [`test_cuda_parity`](tests/test_cuda_parity.cpp) | GPU vs CPU: occupancy flips < 0.1%, changed render pixels < 1%. Built only when CUDA is present. |

Run: `ctest --test-dir build --output-on-failure`. Artifacts in
`build/test_artifacts/`.

## Roadmap

| Week | Plan |
| --- | --- |
| 1 | ✅ CPU SfS + ray-march, file formats, synthetic data, tests, interactive web viewer |
| 2 | ✅ CUDA kernels for reconstruction and rendering, `--gpu` flag, parity test |
| 3 | ✅ Grid-resident GPU render loop, CUDA-event + `ncu` profiling ([docs/PERFORMANCE.md](docs/PERFORMANCE.md)); ⬜ texture-memory rewrite |
| 4 | ✅ GPU-backed interactive viewer (`bake_views --gpu`); ⬜ live windowed camera, final demo |

CPU paths stay as the reference the CUDA versions validate against.
