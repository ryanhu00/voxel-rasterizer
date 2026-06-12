#!/usr/bin/env bash
#
# photos_to_voxels.sh — drive the full pipeline from a directory of photos
# (PNG / JPG / JPEG / PPM) to a reconstructed voxel grid and a baked
# interactive viewer. PNG/JPG inputs are converted to PPM automatically.
#
# Usage:
#   scripts/photos_to_voxels.sh <input_dir> <out_dir>
#                               [--masks-dir DIR | --no-mask]
#                               [--grid N] [--gpu]
#                               [-- <make_orbit_cameras flags>]
#
# Example:
#   scripts/photos_to_voxels.sh ~/photos/bottle data/bottle \
#       -- --radius 5 --fov 1.2
#
# Run from the repo root (the script looks for build/ here).

set -euo pipefail

print_help() {
  sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Arguments:
  input_dir          Directory of source photos. Sorted alphabetically; that
                     order becomes the orbit order around the object.
  out_dir            Working directory; will be created. Receives rgb_*.ppm,
                     mask_*.pgm, cameras.txt, voxels.bin, and view/.

Options:
  --masks-dir DIR    Use pre-existing masks from DIR (alphabetical pairing).
                     Masks are converted to PGM via ImageMagick if needed.
                     Skips rembg.
  --no-mask          Don't generate masks. Stops after cameras.txt.
  --grid N           Voxel grid resolution (default 128).
  --gpu              Run reconstruct and bake_views with CUDA (--gpu). Requires
                     a CUDA build of the project.
  --                 Everything after -- is forwarded to make_orbit_cameras
                     (e.g. --radius, --elev, --fov, --target, --start-deg).

Required tools (the script reports the first one it can't find):
  build/make_orbit_cameras  build/reconstruct  build/bake_views
  Python Pillow (pip install pillow) for JPG/PNG EXIF-aware conversion
  rembg + onnxruntime (only if generating masks automatically; pip install)
  convert (ImageMagick; only for --masks-dir with non-PGM masks)
EOF
}

# Write mask_%04d.pgm with the same width/height as rgb_%04d.ppm.
# Transposes when W/H are swapped; otherwise nearest-neighbor resizes.
align_masks_to_rgb() {
  local dir=$1 count=$2
  python3 - "$dir" "$count" <<'PY'
import sys
from pathlib import Path
from PIL import Image
import numpy as np

out_dir = Path(sys.argv[1])
count = int(sys.argv[2])

def pnm_wh(path, magic):
    with open(path, "rb") as f:
        if f.readline().strip() != magic.encode():
            raise ValueError(f"expected {magic} in {path}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        return map(int, line.split())

def load_pgm(path):
    w, h = pnm_wh(path, "P5")
    with open(path, "rb") as f:
        f.readline()
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        f.readline()  # maxval
        arr = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    return arr, w, h

def save_pgm(path, arr):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())

def fit_mask(arr, tw, th):
    h, w = arr.shape
    if w == tw and h == th:
        return arr
    if w == th and h == tw:
        return arr.T
    return np.array(Image.fromarray(arr).resize((tw, th), Image.NEAREST))

for i in range(count):
    rgb = out_dir / f"rgb_{i:04d}.ppm"
    mask = out_dir / f"mask_{i:04d}.pgm"
    if not mask.is_file():
        continue
    tw, th = pnm_wh(rgb, "P6")
    arr, mw, mh = load_pgm(mask)
    fixed = fit_mask(arr, tw, th)
    if (mw, mh) != (tw, th):
        action = "transposed" if mw == th and mh == tw else "resized"
        print(f"  aligned mask_{i:04d} ({action}): {mw}x{mh} -> {tw}x{th}",
              flush=True)
    save_pgm(mask, fixed)
PY
}

if [ $# -lt 2 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  print_help; exit 0
fi

IN_DIR=$1; OUT_DIR=$2; shift 2

MASKS_DIR=""
SKIP_MASK=0
GRID_N=128
USE_GPU=0
CAMERA_FLAGS=()

while [ $# -gt 0 ]; do
  case $1 in
    --masks-dir) MASKS_DIR=$2; shift 2;;
    --no-mask)   SKIP_MASK=1; shift;;
    --grid)      GRID_N=$2; shift 2;;
    --gpu)       USE_GPU=1; shift;;
    --)          shift; CAMERA_FLAGS=("$@"); break;;
    -h|--help)   print_help; exit 0;;
    *)           echo "unknown arg: $1" >&2; exit 1;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
die()  { echo "error: $*" >&2; exit 1; }

# ---- 0) Sanity check the tools we'll need ----------------------------------
[ -x build/make_orbit_cameras ] || die "build/make_orbit_cameras not found; run cmake --build build first"
[ -x build/reconstruct       ] || die "build/reconstruct not found; run cmake --build build first"
[ -x build/bake_views        ] || die "build/bake_views not found; run cmake --build build first"

HAVE_CONVERT=0;   have convert   && HAVE_CONVERT=1

# JPG/PNG -> PPM with EXIF orientation applied (matches phone preview).
# Raw jpegtopnm ignores EXIF and breaks mask/RGB alignment with rembg.
photo_to_ppm() {
  python3 - "$1" "$2" <<'PY'
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if src.suffix.lower() in (".ppm", ".pnm"):
    shutil.copy2(src, dst)
    raise SystemExit(0)

img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
w, h = img.size
with open(dst, "wb") as f:
    f.write(f"P6\n{w} {h}\n255\n".encode())
    f.write(img.tobytes())
PY
}

# Convert one image to .ppm at $2.
convert_rgb() {
  local src=$1 dst=$2
  local ext_lc; ext_lc=$(printf '%s' "${src##*.}" | tr '[:upper:]' '[:lower:]')
  case "$ext_lc" in
    ppm|pnm) cp "$src" "$dst";;
    png|jpg|jpeg)
      python3 -c "import PIL" >/dev/null 2>&1 \
        || die "need Pillow for JPG/PNG (pip install pillow)"
      photo_to_ppm "$src" "$dst";;
    *) die "unsupported image format: .$ext_lc ($src)";;
  esac
}

# Convert one mask file to .pgm at $2.
convert_mask() {
  local src=$1 dst=$2
  local ext_lc; ext_lc=$(printf '%s' "${src##*.}" | tr '[:upper:]' '[:lower:]')
  if [ "$ext_lc" = "pgm" ]; then
    cp "$src" "$dst"
  else
    [ $HAVE_CONVERT -eq 1 ] || die "need ImageMagick to convert mask format .$ext_lc -> pgm"
    convert "$src" -colorspace gray "$dst"
  fi
}

# ---- 1) Inputs -> rgb_NNNN.ppm ---------------------------------------------
[ -d "$IN_DIR" ] || die "input dir not found: $IN_DIR"
mkdir -p "$OUT_DIR"
# Clean previous run's outputs to keep counts consistent.
rm -f "$OUT_DIR"/rgb_*.ppm "$OUT_DIR"/mask_*.pgm "$OUT_DIR"/cameras.txt

echo "[1/5] Converting inputs -> PPM"
i=0
while IFS= read -r f; do
  printf -v out "%s/rgb_%04d.ppm" "$OUT_DIR" "$i"
  convert_rgb "$f" "$out"
  i=$((i+1))
done < <(find "$IN_DIR" -maxdepth 1 -type f \
           \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
              -o -iname '*.ppm' -o -iname '*.pnm' \) | sort)

N=$i
[ "$N" -ge 4 ] || die "found only $N images; need at least 4 (24+ recommended)"
echo "      -> $N -> $OUT_DIR/rgb_*.ppm"

# ---- 2) Masks --------------------------------------------------------------
if [ "$SKIP_MASK" -eq 1 ]; then
  echo "[2/5] --no-mask set, skipping masks"
elif [ -n "$MASKS_DIR" ]; then
  echo "[2/5] Using masks from $MASKS_DIR"
  j=0
  while IFS= read -r f; do
    printf -v out "%s/mask_%04d.pgm" "$OUT_DIR" "$j"
    convert_mask "$f" "$out"
    j=$((j+1))
  done < <(find "$MASKS_DIR" -maxdepth 1 -type f | sort)
  [ "$j" -eq "$N" ] || \
    echo "      warning: $N rgb but $j masks; reconstruct will pair by index up to min" >&2
  align_masks_to_rgb "$OUT_DIR" "$N"
  echo "      -> $j -> $OUT_DIR/mask_*.pgm"
else
  echo "[2/5] Generating masks with rembg"
  # rembg on the same EXIF-corrected RGB as step 1 (not raw JPEG bytes).
  python3 -c "import rembg" >/dev/null 2>&1 \
    || die "rembg not importable (pip install rembg onnxruntime). Use --masks-dir or --no-mask."

  in_list=$(mktemp)
  trap 'rm -f "$in_list"' EXIT
  j=0
  while IFS= read -r f; do
    echo "$f" >> "$in_list"
    j=$((j+1))
  done < <(find "$IN_DIR" -maxdepth 1 -type f \
             \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)

  python3 - "$OUT_DIR" "$in_list" <<'PY'
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove

out_dir = Path(sys.argv[1])
ins = [Path(l.strip()) for l in open(sys.argv[2]) if l.strip()]

def ppm_wh(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"P6":
            raise ValueError(f"expected P6 in {path}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        return map(int, line.split())

def save_pgm(path, arr):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())

def fit_alpha(alpha, tw, th):
    h, w = alpha.shape
    if w == tw and h == th:
        return alpha
    if w == th and h == tw:
        return alpha.T
    return np.array(Image.fromarray(alpha).resize((tw, th), Image.NEAREST))

def load_oriented_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")

session = new_session()
for i, src in enumerate(ins):
    rgb_ppm = out_dir / f"rgb_{i:04d}.ppm"
    tw, th = ppm_wh(rgb_ppm)
    img = load_oriented_rgb(src)
    if img.size != (tw, th):
        raise SystemExit(
            f"error: {src.name} oriented size {img.size} != {rgb_ppm.name} {tw}x{th}"
        )
    buf = BytesIO()
    img.save(buf, format="PNG")
    cut = remove(buf.getvalue(), session=session)
    rgba = Image.open(BytesIO(cut)).convert("RGBA")
    if rgba.size != img.size:
        rgba = rgba.resize(img.size, Image.NEAREST)
    alpha = fit_alpha(np.array(rgba.split()[3]), tw, th)
    mask = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    out = out_dir / f"mask_{i:04d}.pgm"
    save_pgm(out, mask)
    print(f"  rembg {i+1}/{len(ins)}: {src.name} -> {out.name} ({tw}x{th})",
          flush=True)
PY

  echo "      -> $j -> $OUT_DIR/mask_*.pgm"
fi

# ---- 3) cameras.txt --------------------------------------------------------
echo "[3/5] Generating cameras.txt"
if [ "${#CAMERA_FLAGS[@]}" -eq 0 ]; then
  echo "      (no flags after --; using make_orbit_cameras defaults)"
fi
build/make_orbit_cameras --in "$OUT_DIR" "${CAMERA_FLAGS[@]}"

if [ "$SKIP_MASK" -eq 1 ]; then
  echo ""
  echo "Stopped after cameras.txt (--no-mask). Add masks and re-run without --no-mask"
  echo "to continue to reconstruct + bake."
  exit 0
fi

# ---- 4) Reconstruct --------------------------------------------------------
GPU_FLAG=()
[ "$USE_GPU" -eq 1 ] && GPU_FLAG=(--gpu)
echo "[4/5] Reconstructing voxel grid (${GRID_N}^3$([ "$USE_GPU" -eq 1 ] && echo ', GPU'))"
build/reconstruct --in "$OUT_DIR" --out "$OUT_DIR/voxels.bin" --grid "$GRID_N" \
                  "${GPU_FLAG[@]}"

# ---- 5) Bake views ---------------------------------------------------------
echo "[5/5] Baking interactive viewer$([ "$USE_GPU" -eq 1 ] && echo ' (GPU)')"
build/bake_views --voxels "$OUT_DIR/voxels.bin" --out "$OUT_DIR/view" \
                 "${GPU_FLAG[@]}"
