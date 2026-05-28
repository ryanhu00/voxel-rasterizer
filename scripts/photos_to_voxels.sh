#!/usr/bin/env bash
#
# photos_to_voxels.sh — drive the full pipeline from a directory of photos
# (PNG / JPG / JPEG / PPM) to a reconstructed voxel grid and a baked
# interactive viewer. PNG/JPG inputs are converted to PPM automatically.
#
# Usage:
#   scripts/photos_to_voxels.sh <input_dir> <out_dir>
#                               [--masks-dir DIR | --no-mask]
#                               [--grid N]
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
  --                 Everything after -- is forwarded to make_orbit_cameras
                     (e.g. --radius, --elev, --fov, --target, --start-deg).

Required tools (the script reports the first one it can't find):
  build/make_orbit_cameras  build/reconstruct  build/bake_views
  pngtopnm / jpegtopnm (netpbm) OR convert (ImageMagick)
  rembg (only if generating masks automatically)
  convert (ImageMagick; only if generating masks or converting non-PGM masks)
EOF
}

if [ $# -lt 2 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  print_help; exit 0
fi

IN_DIR=$1; OUT_DIR=$2; shift 2

MASKS_DIR=""
SKIP_MASK=0
GRID_N=128
CAMERA_FLAGS=()

while [ $# -gt 0 ]; do
  case $1 in
    --masks-dir) MASKS_DIR=$2; shift 2;;
    --no-mask)   SKIP_MASK=1; shift;;
    --grid)      GRID_N=$2; shift 2;;
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

HAVE_PNGTOPNM=0;  have pngtopnm  && HAVE_PNGTOPNM=1
HAVE_JPEGTOPNM=0; have jpegtopnm && HAVE_JPEGTOPNM=1
HAVE_CONVERT=0;   have convert   && HAVE_CONVERT=1

# Convert one image to .ppm at $2.
convert_rgb() {
  local src=$1 dst=$2
  local ext_lc; ext_lc=$(printf '%s' "${src##*.}" | tr '[:upper:]' '[:lower:]')
  case "$ext_lc" in
    ppm|pnm) cp "$src" "$dst";;
    png)
      if [ $HAVE_PNGTOPNM -eq 1 ]; then pngtopnm "$src" > "$dst"
      elif [ $HAVE_CONVERT -eq 1 ]; then convert "$src" "$dst"
      else die "no PNG converter; install netpbm or imagemagick"; fi;;
    jpg|jpeg)
      if [ $HAVE_JPEGTOPNM -eq 1 ]; then jpegtopnm "$src" > "$dst"
      elif [ $HAVE_CONVERT -eq 1 ]; then convert "$src" "$dst"
      else die "no JPEG converter; install netpbm or imagemagick"; fi;;
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
  echo "      -> $j -> $OUT_DIR/mask_*.pgm"
else
  echo "[2/5] Generating masks with rembg"
  # We deliberately use the rembg Python API instead of its CLI: rembg's CLI
  # has a sprawling set of optional deps (filetype, watchdog, click, fastapi,
  # ...) that pip-install-rembg doesn't reliably pull in, and any one of them
  # missing makes the CLI unrunnable. The library API only needs the core
  # inference deps that pip definitely installs.
  python3 -c "import rembg" >/dev/null 2>&1 \
    || die "rembg not installed (pip install rembg). Use --masks-dir or --no-mask."
  have convert || die "ImageMagick (convert) needed to extract alpha. apt install imagemagick."

  # Collect input file list, then hand it to a Python helper that loads rembg
  # once (model load is the expensive part) and segments each image.
  in_list=$(mktemp); out_list=$(mktemp)
  trap 'rm -f "$in_list" "$out_list"' EXIT
  j=0
  while IFS= read -r f; do
    printf -v out "%s/.mask_alpha_%04d.png" "$OUT_DIR" "$j"
    echo "$f"   >> "$in_list"
    echo "$out" >> "$out_list"
    j=$((j+1))
  done < <(find "$IN_DIR" -maxdepth 1 -type f \
             \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)

  python3 - "$in_list" "$out_list" <<'PY'
import sys
from rembg import remove, new_session
session = new_session()  # default u2net
with open(sys.argv[1]) as f: ins  = [l.strip() for l in f if l.strip()]
with open(sys.argv[2]) as f: outs = [l.strip() for l in f if l.strip()]
for i, (src, dst) in enumerate(zip(ins, outs)):
    with open(src, 'rb') as h: data = h.read()
    cut = remove(data, session=session)
    with open(dst, 'wb') as h: h.write(cut)
    print(f"  rembg {i+1}/{len(ins)}: {src} -> {dst}", flush=True)
PY

  # Extract alpha -> 8-bit PGM mask.
  for ((k=0; k<j; k++)); do
    printf -v cut  "%s/.mask_alpha_%04d.png" "$OUT_DIR" "$k"
    printf -v mask "%s/mask_%04d.pgm"        "$OUT_DIR" "$k"
    convert "$cut" -alpha extract -threshold 50% "$mask"
    rm -f "$cut"
  done
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
echo "[4/5] Reconstructing voxel grid (${GRID_N}^3)"
build/reconstruct --in "$OUT_DIR" --out "$OUT_DIR/voxels.bin" --grid "$GRID_N"

# ---- 5) Bake views ---------------------------------------------------------
echo "[5/5] Baking interactive viewer"
build/bake_views --voxels "$OUT_DIR/voxels.bin" --out "$OUT_DIR/view"

echo ""
echo "Done. To view:"
echo "  cd $OUT_DIR/view && python3 -m http.server 8000"
echo "Then open: http://localhost:8000/viewer.html"
