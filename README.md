# voxel-rasterizer

A GPU-accelerated multi-view voxel reconstruction and rasterization pipeline
(CS179 project, Ryan Hu & Samuel Xie). Given a set of calibrated images of an
object, the system

1. **reconstructs** a 3D voxel occupancy + color grid (shape-from-silhouette),
2. **renders** the grid from arbitrary novel viewpoints via GPU ray-marching.

This repository currently contains the **Week 1 CPU prototype** that establishes
the data layouts, camera math, file formats, and reference implementations the
CUDA kernels in subsequent weeks will replace.

## Repository layout

```
include/voxr/          Public headers (math, camera, voxel_grid, scene, etc.)
src/                   CPU library implementations
apps/                  CLI tools (synth_dataset, reconstruct, render)
tests/                 Self-contained tests, runnable via ctest
data/                  Generated datasets (git-ignored)
```

The CPU code is organized so that each kernel-like loop (per-voxel SfS and
per-pixel ray-march) becomes a near-1:1 translation into CUDA in Week 2.

## Build

Requires a C++17 compiler and CMake ≥ 3.16.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Three CLI executables are produced in `build/`:

| Binary | Purpose |
| --- | --- |
| `synth_dataset` | Render an analytic shape (sphere / cube / dumbbell) from a ring of cameras to produce ground-truth RGB images, silhouette masks, and a camera dataset. |
| `reconstruct` | Run CPU shape-from-silhouette on a dataset to produce a voxel grid. |
| `render` | Render a saved voxel grid to PPM, either a single image or an orbit turntable. |

## End-to-end demo (sphere)

```bash
# 1. Synthesize 24 calibrated views of a unit sphere
build/synth_dataset --out data/sphere --shape sphere --views 24 --res 256 256

# 2. Reconstruct a 128^3 voxel grid with color fusion
build/reconstruct --in data/sphere --out data/sphere/voxels.bin --grid 128

# 3. Render the reconstruction from a novel viewpoint
build/render --voxels data/sphere/voxels.bin --out data/sphere/novel.ppm \
             --eye 1.6 1.0 1.6 --target 0 0 0 --up 0 1 0 --res 512 512

# 4. (Optional) Generate a 36-frame orbit turntable
build/render --voxels data/sphere/voxels.bin --orbit data/sphere/orbit \
             --views 36 --radius 3 --res 512 512
```

You can swap `--shape cube` or `--shape dumbbell` for a more interesting test
of the visual hull's behavior on concavity.

## File formats

- **Color images:** Netpbm PPM (`P6`, 8-bit RGB).
- **Silhouette masks:** Netpbm PGM (`P5`, 8-bit; ≥128 = foreground).
- **Camera dataset (`cameras.txt`):** human-readable text, one record per
  camera with intrinsics, world-space position, and rotation matrix. Schema
  is documented at the top of `include/voxr/camera.hpp`.
- **Voxel grid (`*.bin`):** little-endian binary, `VOXG` magic, version 1.
  Contains dimensions, world origin, voxel size, then four uint8 channels
  (`occupancy`, `r`, `g`, `b`) in linear order
  `idx = x + nx*(y + ny*z)`. Layout details in `include/voxr/voxel_grid.hpp`.

## Conventions

- **Camera frame:** OpenCV-style. +X right, +Y down in image, +Z forward into
  the scene. Right-handed.
- **World frame:** arbitrary right-handed. The CLI defaults assume +Y up; the
  `--up` flag lets you override per-camera.
- **Projection:** `pixel = K · R · (X_world − t_world)`, divided by the
  resulting `z`. See `Camera::project` for the precise formula.

## Roadmap (subsequent weeks)

| Week | Plan |
| --- | --- |
| 1 | **(this code)** CPU SfS + ray-march, file formats, synthetic data, tests |
| 2 | CUDA kernels for reconstruction and rendering; integration with real images |
| 3 | Color fusion improvements, memory/traversal optimization, benchmarks |
| 4 | Interactive camera, scalability + perf tuning, final demo and report |

The CPU paths in this repo will be kept as the reference implementations the
CUDA versions are validated against in CI-style tests.
