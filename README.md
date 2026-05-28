# voxel-rasterizer

GPU-accelerated multi-view voxel reconstruction + rasterization
(CS179, Ryan Hu & Samuel Xie). Given calibrated images, reconstruct a 3D
occupancy + color grid via shape-from-silhouette, then ray-march novel views.

This repo is the **Week 1 CPU prototype**. Each kernel-like loop (per-voxel
SfS, per-pixel ray-march) is structured for a near-1:1 CUDA port in Week 2.

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

| Binary | Purpose |
| --- | --- |
| `synth_dataset` | Render an analytic shape (sphere/cube/dumbbell) from a ring of cameras → RGB + masks + `cameras.txt`. |
| `reconstruct` | CPU shape-from-silhouette → voxel grid. |
| `render` | Render a voxel grid to PPM (single view or orbit). |
| `bake_views` | Pre-render an azimuth × elevation × radius grid + self-contained HTML viewer (drag = orbit, scroll = zoom). |
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

Run: `ctest --test-dir build --output-on-failure`. Artifacts in
`build/test_artifacts/`.

## Roadmap

| Week | Plan |
| --- | --- |
| 1 | **(this code)** CPU SfS + ray-march, file formats, synthetic data, tests, interactive web viewer |
| 2 | CUDA kernels for reconstruction and rendering; integration with real images |
| 3 | Color fusion improvements, memory/traversal optimization, benchmarks |
| 4 | Interactive camera, scalability + perf tuning, final demo and report |

CPU paths stay as the reference the CUDA versions validate against.
