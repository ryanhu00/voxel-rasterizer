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

The CLI executables produced in `build/`:

| Binary | Purpose |
| --- | --- |
| `synth_dataset` | Render an analytic shape (sphere / cube / dumbbell) from a ring of cameras to produce ground-truth RGB images, silhouette masks, and a camera dataset. |
| `reconstruct` | Run CPU shape-from-silhouette on a dataset to produce a voxel grid. |
| `render` | Render a saved voxel grid to PPM, either a single image or an orbit turntable. |
| `bake_views` | Pre-render a 3D grid of viewpoints (azimuth × elevation × radius) plus a self-contained HTML viewer that snaps to the nearest baked view as you drag/zoom. |
| `make_orbit_cameras` | Generate a `cameras.txt` for a directory of your own photos taken on a ring orbit (the missing piece for the "bring your own PNGs" workflow). |

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

# 4a. Convert the PPM orbit frames to PNG (needs `netpbm`)
for f in data/sphere/orbit/*.ppm; do pnmtopng "$f" > "${f%.ppm}.png"; done

# 4b. Stitch the PNGs into an mp4 (needs `ffmpeg`)
ffmpeg -y -framerate 24 -i data/sphere/orbit/frame_%04d.png \
       -c:v libx264 -pix_fmt yuv420p data/sphere/orbit.mp4

# 5. (Optional) Bake an interactive 3D viewer
#    Drag = orbit (azimuth + elevation), scroll = zoom, R = reset.
build/bake_views --voxels data/sphere/voxels.bin --out data/sphere/view \
                 --azim 24 --elev 7 --radii 3 --res 256 256
cd data/sphere/view && python3 -m http.server 8000
# then open http://localhost:8000/viewer.html
```

You can swap `--shape cube` or `--shape dumbbell` for a more interesting test
of the visual hull's behavior on concavity.

## Bring your own photos

For real images you supply yourself, the pipeline needs three things per view:
the RGB photo as PPM, a silhouette mask as PGM, and an entry in `cameras.txt`
describing the pose. Two of those are mechanical conversions; the third is the
only step the project couldn't infer for you, so `make_orbit_cameras` writes it
from a description of the shoot.

The shoot has to fit a ring-orbit model: camera moves on a horizontal circle
of known radius around the object, at roughly constant elevation, with the
photos saved in shooting order (so alphabetical filename order matches the
angular order around the ring). If that's not your setup, you'll need to fall
back to a structure-from-motion tool (e.g. COLMAP) to produce poses, and write
a `cameras.txt` from those instead — the format is documented at the top of
[include/voxr/camera.hpp](include/voxr/camera.hpp).

```bash
# 0. Put your photos in a folder, named so alphabetical = shooting order
mkdir -p data/myobj/png && cp /path/to/your/photos/*.png data/myobj/png/

# 1. Convert PNG -> PPM (needs `netpbm`); rename to rgb_NNNN.ppm
i=0; for f in data/myobj/png/*.png; do
  printf -v out "data/myobj/rgb_%04d.ppm" "$i"
  pngtopnm "$f" > "$out"
  i=$((i+1))
done

# 2. Provide silhouette masks as data/myobj/mask_NNNN.pgm (255 = foreground).
#    If you don't already have masks, a quick way for photos against a flat
#    light background is ImageMagick (tune the threshold per shoot):
#      i=0; for f in data/myobj/png/*.png; do
#        printf -v out "data/myobj/mask_%04d.pgm" "$i"
#        convert "$f" -colorspace Gray -threshold 80% -negate "$out"
#        i=$((i+1))
#      done
#    For tougher backgrounds use a segmentation tool (rembg, SAM, etc.) and
#    write the resulting masks out as 8-bit PGM.

# 3. Generate cameras.txt for the ring orbit
#    --radius is the camera distance to the object in your world units.
#    --elev is the (constant) elevation in degrees.
#    --start-deg / --cw control which photo corresponds to which angle.
build/make_orbit_cameras --in data/myobj \
                         --radius 3.0 --elev 15 --fov 0.9 \
                         --target 0 0 0 --up 0 1 0

# 4. Reconstruct + bake + view (same as the sphere demo)
build/reconstruct --in data/myobj --out data/myobj/voxels.bin --grid 128
build/bake_views --voxels data/myobj/voxels.bin --out data/myobj/view
cd data/myobj/view && python3 -m http.server 8000
# then open http://localhost:8000/viewer.html
```

If the reconstruction comes out too sparse, your `--radius` / `--elev` /
`--start-deg` likely don't match how the photos were actually taken — shape-
from-silhouette is unforgiving about pose errors. Start by reconstructing with
`--min-views N` lower than `num_photos` to confirm the geometry roughly lines
up, then tighten.

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

## Tests

The repo ships a minimal header-only harness in
[tests/test_harness.hpp](tests/test_harness.hpp) (no GoogleTest, no Catch2 —
just `VOXR_EXPECT` / `VOXR_EXPECT_NEAR` macros that count failures and return
the count from `main`, so CTest picks up the pass/fail). Three binaries are
wired into `ctest`:

| Test | Covers | What it checks |
| --- | --- | --- |
| `test_camera` | [tests/test_camera.cpp](tests/test_camera.cpp) | `Camera::project` ↔ `unproject_ray` round-trip on a `look_at` camera; the world origin projects to the principal point; world +Y projects above center (image +Y is down); world +X projects to the right of center; points behind the camera fail to project. Catches sign / handedness bugs in the projection math. |
| `test_voxel_grid` | [tests/test_voxel_grid.cpp](tests/test_voxel_grid.cpp) | Linear indexing matches the documented `x + nx*(y + ny*z)` layout; `voxel_center` ↔ `world_to_grid` are inverses; the `VOXG` binary format round-trips dims, origin, voxel_size, occupancy, and all three color channels through save+load. Catches layout or serialization regressions before they bleed into reconstructions on disk. |
| `test_pipeline` | [tests/test_pipeline.cpp](tests/test_pipeline.cpp) | End-to-end integration: synthesizes a small 12-view sphere dataset, reconstructs a 64³ grid from the masks, compares the occupancy against the analytic ground truth (requires **recall > 0.95** of truly-interior voxels and **extra/gt < 0.40** for visual-hull bloat), then ray-marches a novel viewpoint and checks the rendered image has a sensible foreground fraction (5%–95%). This is the test the CUDA port will be validated against. |

Run them with:

```bash
ctest --test-dir build --output-on-failure
```

Test artifacts (the round-tripped grid, the sphere dataset, the rendered novel
view) are written under `build/test_artifacts/` so you can eyeball them after a
run.

## Roadmap (subsequent weeks)

| Week | Plan |
| --- | --- |
| 1 | **(this code)** CPU SfS + ray-march, file formats, synthetic data, tests |
| 2 | CUDA kernels for reconstruction and rendering; integration with real images |
| 3 | Color fusion improvements, memory/traversal optimization, benchmarks |
| 4 | Interactive camera, scalability + perf tuning, final demo and report |

The CPU paths in this repo will be kept as the reference implementations the
CUDA versions are validated against in CI-style tests.
