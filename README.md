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

### Performance

GPU vs CPU for both pipeline stages, plus `ncu` kernel profiles. The CPU paths
(16 `std::thread` workers) are the reference; the CUDA kernels reproduce their
output bit-for-bit on the parity test.

**Setup:** NVIDIA RTX A5000 (sm_86), CUDA 12.5, 16-thread CPU baseline. Sphere
dataset, 24 views. Reproduce with the commands in each section.

#### Headline numbers

| Workload | CPU | GPU | Speedup |
| --- | --- | --- | --- |
| Reconstruct 256³ (kernel only) | — | **13.9 ms** | — |
| Reconstruct 256³ (end-to-end wall) | 2.01 s | 0.52 s | ~3.9× |
| Render 1024² (kernel only) | — | **1.70 ms** (≈480 fps) | — |
| Render orbit 36×512² (wall) | 1.85 s | 0.57 s | ~3.2× |

The end-to-end GPU times include one-time CUDA context init (~0.3 s) and disk
I/O shared with the CPU runs, so they *understate* the compute speedup. The
kernel-only times below are the honest device cost.

#### Isolated timing (CUDA events)

`render --bench` keeps the grid resident on the device and times the kernel and
readback separately, so the per-frame cost is real interactive cost — not
polluted by the grid upload.

```
build/render --voxels voxels.bin --res 1024 1024 --bench 200
  grid upload (1x) :    9.325 ms      # paid once
  kernel    avg/min:    2.095 / 1.717 ms   (477 / 582 fps)
  readback  avg    :    0.372 ms
  frame (kernel+rb):    2.466 ms      (405 fps)
```

```
VOXR_CUDA_TIMING=1 build/reconstruct --in data/sphere --grid 256 --gpu
  [voxr] reconstruct_kernel: 17.1 ms (256^3 voxels, 24 views)
```

**Why the resident loop matters:** re-uploading the 64 MB grid every frame
(what the one-shot `render_cuda` does) costs ~9 ms — more than the 1.7 ms
kernel. `CudaVoxelRenderer` uploads once, which is what makes the orbit and the
interactive viewer (`bake_views --gpu`) a true GPU render loop. This is the
fix for the "single small render favors the CPU" caveat: amortized over frames,
the GPU runs the trace at ~480 fps.

#### `ncu` kernel profiles

```
ncu --launch-count 1 --kernel-name <k> \
    --section SpeedOfLight --section Occupancy --section WarpStateStats <cmd>
```

| Metric | `render_kernel` | `reconstruct_kernel` |
| --- | --- | --- |
| Duration | 1.70 ms | 13.89 ms |
| Compute (SM) throughput | 79.0 % | 75.1 % |
| Memory throughput | 27.6 % | **82.8 %** |
| DRAM throughput | 1.1 % | 0.4 % |
| Achieved / theoretical occupancy | 93.4 % / 100 % | 98.5 % / 100 % |

`render_kernel` warp-issue stalls (cycles per issued instr, top reasons):
`not_selected` 3.58, `long_scoreboard` 2.82, `wait` 2.60, `lg_throttle` 0.00.

##### Reading it

- **Both kernels hit ~full occupancy** — the launch geometry (8³ voxel blocks,
  16×16 pixel tiles) saturates the SMs. No occupancy left on the table.
- **`render_kernel` is compute-bound** (SM 79 % vs mem 28 %, DRAM ~1 %). The
  DDA arithmetic dominates; the grid stays hot in L2 because spatially coherent
  rays in a 16×16 tile reuse the same voxel cache lines, so DRAM traffic is
  negligible. `not_selected` being the top stall means there are *plenty* of
  eligible warps — the scheduler is busy, which is what you want. The secondary
  `long_scoreboard` stalls are the occupancy/color global loads.
- **`reconstruct_kernel` is bound by the memory pipeline** (82.8 %), not DRAM
  (0.4 %). Each voxel does a bilinear gather into every view's mask buffer;
  neighbouring voxels project to *nearby but non-contiguous* pixels, so the
  loads are scattered through L1/L2 rather than coalesced. The data is small
  enough to stay cached (low DRAM), so the cost is the gather pattern itself.

#### What to optimize next

The proposal names texture memory and traversal optimization. The profile says
where each pays off:

- **Textures help reconstruction most.** Binding masks/images as
  `cudaTextureObject_t` (the plan in [src/reconstruct_cpu.cpp](src/reconstruct_cpu.cpp))
  replaces the scattered manual bilinear gather with a single hardware-filtered
  fetch through the 2D texture cache — directly attacking the 82.8 % memory-pipe
  bottleneck. Occupancy is already maxed, so this is the real lever here.
- **`render_kernel` has less to gain from memory work** — it's SM-bound with
  ~1 % DRAM. A 3D occupancy texture would trim the `long_scoreboard` stalls a
  little, but the ceiling is the DDA arithmetic. The bigger win at very deep
  grids would be reducing the per-warp step-count variance (persistent threads),
  not memory.

Net: reconstruction is the better optimization target, and texture memory is
the right tool — the profile confirms the proposal's instinct rather than
guessing.

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
| 3 | ✅ Grid-resident GPU render loop, CUDA-event + `ncu` profiling (see Performance above); ⬜ texture-memory rewrite |
| 4 | ✅ GPU-backed interactive viewer (`bake_views --gpu`); ⬜ live windowed camera, final demo |

CPU paths stay as the reference the CUDA versions validate against.
