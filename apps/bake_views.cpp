// bake_views: pre-render a 3D grid of viewpoints (azimuth x elevation x radius)
// around a voxel grid and emit a manifest plus a self-contained HTML viewer.
//
// The viewer is a single static page that loads the manifest, parses the PPM
// frames in JavaScript, and snaps to the nearest baked viewpoint as the user
// drags the mouse (drag = orbit, vertical drag = elevation, scroll = zoom).
//
// Usage:
//   bake_views --voxels FILE --out DIR
//              [--res W H] [--fov RAD]
//              [--azim N] [--elev N] [--elev-range MIN_DEG MAX_DEG]
//              [--radii N] [--radius-range MIN MAX]
//              [--target X Y Z] [--up X Y Z] [--no-shading]

#include "voxr/camera.hpp"
#include "voxr/image.hpp"
#include "voxr/render_cpu.hpp"
#include "voxr/voxel_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string voxels_path;
    std::string out_dir = "bake";
    int   width  = 256;
    int   height = 256;
    float fov_y  = 0.9f;
    int   azim_count   = 24;
    int   elev_count   = 7;
    float elev_min_deg = -45.f;
    float elev_max_deg =  45.f;
    int   radius_count = 3;
    float radius_min = 0.f;     // 0 -> derive from grid
    float radius_max = 0.f;
    float tx = 0.f, ty = 0.f, tz = 0.f;
    bool  target_set = false;
    float ux = 0.f, uy = 1.f, uz = 0.f;
    bool  no_shading = false;
};

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                std::cerr << "missing value for " << s << "\n"; std::exit(1);
            }
        };
        if      (s == "--voxels")       { need(1); a.voxels_path = argv[++i]; }
        else if (s == "--out")          { need(1); a.out_dir     = argv[++i]; }
        else if (s == "--res")          { need(2); a.width  = std::atoi(argv[++i]);
                                                   a.height = std::atoi(argv[++i]); }
        else if (s == "--fov")          { need(1); a.fov_y = std::atof(argv[++i]); }
        else if (s == "--azim")         { need(1); a.azim_count = std::atoi(argv[++i]); }
        else if (s == "--elev")         { need(1); a.elev_count = std::atoi(argv[++i]); }
        else if (s == "--elev-range")   { need(2); a.elev_min_deg = std::atof(argv[++i]);
                                                   a.elev_max_deg = std::atof(argv[++i]); }
        else if (s == "--radii")        { need(1); a.radius_count = std::atoi(argv[++i]); }
        else if (s == "--radius-range") { need(2); a.radius_min = std::atof(argv[++i]);
                                                   a.radius_max = std::atof(argv[++i]); }
        else if (s == "--target")       { need(3); a.tx = std::atof(argv[++i]);
                                                   a.ty = std::atof(argv[++i]);
                                                   a.tz = std::atof(argv[++i]);
                                                   a.target_set = true; }
        else if (s == "--up")           { need(3); a.ux = std::atof(argv[++i]);
                                                   a.uy = std::atof(argv[++i]);
                                                   a.uz = std::atof(argv[++i]); }
        else if (s == "--no-shading")   { a.no_shading = true; }
        else if (s == "-h" || s == "--help") {
            std::cout <<
              "Usage: bake_views --voxels FILE --out DIR\n"
              "                  [--res W H] [--fov RAD]\n"
              "                  [--azim N] [--elev N] [--elev-range MIN_DEG MAX_DEG]\n"
              "                  [--radii N] [--radius-range MIN MAX]\n"
              "                  [--target X Y Z] [--up X Y Z] [--no-shading]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << s << "\n"; return false;
        }
    }
    if (a.voxels_path.empty()) {
        std::cerr << "--voxels FILE is required\n"; return false;
    }
    if (a.azim_count   < 1) { std::cerr << "--azim must be >= 1\n";   return false; }
    if (a.elev_count   < 1) { std::cerr << "--elev must be >= 1\n";   return false; }
    if (a.radius_count < 1) { std::cerr << "--radii must be >= 1\n";  return false; }
    return true;
}

// The viewer is a single-page app: it fetches manifest.json, decodes the PPMs
// in JS, and snaps to the nearest baked viewpoint as the user drags / scrolls.
// Kept as a raw string literal so `bake_views` produces a fully self-contained
// output directory (frames + manifest + viewer).
constexpr const char* kViewerHTML = R"VIEWER(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>voxel-rasterizer viewer</title>
  <style>
    html, body { margin: 0; height: 100%; background: #111; color: #ddd;
                 font-family: system-ui, -apple-system, sans-serif;
                 overflow: hidden; }
    #wrap { display: flex; flex-direction: column; height: 100vh; }
    #header { padding: 8px 12px; background: #1a1a1a; border-bottom: 1px solid #333;
              font-size: 13px; display: flex; justify-content: space-between;
              align-items: center; flex-wrap: wrap; gap: 8px; }
    #stage { flex: 1; display: flex; align-items: center; justify-content: center;
             background: #000; cursor: grab; user-select: none; touch-action: none; }
    #stage.dragging { cursor: grabbing; }
    canvas { image-rendering: pixelated; max-width: 100%; max-height: 100%;
             display: block; }
    #status { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              opacity: 0.7; margin-left: 8px; }
    .badges { display: flex; gap: 6px; flex-wrap: wrap; }
    .badge { background: #2a2a2a; padding: 2px 8px; border-radius: 3px;
             font-size: 12px; }
    #spinner { display: inline-block; width: 10px; height: 10px;
               border: 2px solid #555; border-top-color: #ddd;
               border-radius: 50%; animation: spin 0.8s linear infinite;
               vertical-align: middle; margin-left: 6px; visibility: hidden; }
    #spinner.on { visibility: visible; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div id="wrap">
    <div id="header">
      <div>
        <strong>voxel viewer</strong>
        <span id="status">loading manifest…</span>
        <span id="spinner"></span>
      </div>
      <div class="badges">
        <span class="badge">drag = orbit</span>
        <span class="badge">scroll = zoom</span>
        <span class="badge">R = reset</span>
      </div>
    </div>
    <div id="stage"><canvas id="cv"></canvas></div>
  </div>

<script>
"use strict";

const stage   = document.getElementById('stage');
const canvas  = document.getElementById('cv');
const ctx     = canvas.getContext('2d');
const statusE = document.getElementById('status');
const spinner = document.getElementById('spinner');

let manifest = null;
let curA = 0, curE = 0, curR = 0;
let pending = 0;
const cache = new Map();

function setStatus(s) { statusE.textContent = s; }
function setBusy(on)  { spinner.classList.toggle('on', !!on); }

// PPM (P6) decoder. Tolerates comment lines in the header.
function parsePPM(buf) {
  const u8 = new Uint8Array(buf);
  const td = new TextDecoder();
  let i = 0;
  function skipWS() {
    while (i < u8.length) {
      const c = u8[i];
      if (c === 0x20 || c === 0x0A || c === 0x0D || c === 0x09) { i++; }
      else if (c === 0x23) { while (i < u8.length && u8[i] !== 0x0A) i++; }
      else break;
    }
  }
  function token() {
    skipWS();
    const start = i;
    while (i < u8.length && u8[i] > 0x20 && u8[i] !== 0x23) i++;
    return td.decode(u8.subarray(start, i));
  }
  const magic = token();
  if (magic !== 'P6') throw new Error('not a P6 PPM (magic=' + magic + ')');
  const w = parseInt(token(), 10);
  const h = parseInt(token(), 10);
  parseInt(token(), 10);   // maxval
  i++;                     // exactly one whitespace byte before pixel data
  const out = new Uint8ClampedArray(w * h * 4);
  for (let p = 0, q = i; p < w * h; p++, q += 3) {
    const o = p * 4;
    out[o]     = u8[q];
    out[o + 1] = u8[q + 1];
    out[o + 2] = u8[q + 2];
    out[o + 3] = 255;
  }
  return new ImageData(out, w, h);
}

function frameIndex(a, e, r) {
  return ((a * manifest.elev_count) + e) * manifest.radius_count + r;
}

function framePath(a, e, r) {
  return 'frame_' + String(frameIndex(a, e, r)).padStart(6, '0') + '.ppm';
}

async function loadFrame(a, e, r) {
  const idx = frameIndex(a, e, r);
  const hit = cache.get(idx);
  if (hit) return hit;
  const p = (async () => {
    const resp = await fetch(framePath(a, e, r));
    if (!resp.ok) throw new Error('fetch failed: ' + resp.status);
    return parsePPM(await resp.arrayBuffer());
  })();
  cache.set(idx, p);
  try {
    const img = await p;
    cache.set(idx, img);
    return img;
  } catch (err) {
    cache.delete(idx);
    throw err;
  }
}

let lastDrawn = -1;

async function draw() {
  const want = frameIndex(curA, curE, curR);
  if (want === lastDrawn) return;
  pending++;
  setBusy(true);
  const a = curA, e = curE, r = curR;
  try {
    const img = await loadFrame(a, e, r);
    // Discard if user has since moved past this frame.
    if (frameIndex(curA, curE, curR) !== want) return;
    ctx.putImageData(img, 0, 0);
    lastDrawn = want;
    setStatus(`a=${a}/${manifest.azim_count}  e=${e}/${manifest.elev_count}  r=${r}/${manifest.radius_count}`);
  } catch (err) {
    setStatus('error: ' + err.message);
  } finally {
    pending--;
    if (pending === 0) setBusy(false);
  }
}

function prefetchNeighbors() {
  const A = manifest.azim_count, E = manifest.elev_count, R = manifest.radius_count;
  const ns = [
    [(curA + 1) % A,           curE,                     curR],
    [(curA - 1 + A) % A,       curE,                     curR],
    [curA, Math.min(E - 1, curE + 1),                    curR],
    [curA, Math.max(0,     curE - 1),                    curR],
    [curA, curE,               Math.min(R - 1, curR + 1)],
    [curA, curE,               Math.max(0,     curR - 1)],
  ];
  for (const [a, e, r] of ns) loadFrame(a, e, r).catch(() => {});
}

// -------------------- mouse + touch input --------------------

let dragging = false;
let dragStartX = 0, dragStartY = 0;
let dragStartA = 0, dragStartE = 0;
let pointerId  = null;

function beginDrag(x, y) {
  dragging = true;
  dragStartX = x;  dragStartY = y;
  dragStartA = curA;
  dragStartE = curE;
  stage.classList.add('dragging');
}

function endDrag() {
  if (!dragging) return;
  dragging = false;
  stage.classList.remove('dragging');
  prefetchNeighbors();
}

function moveDrag(x, y) {
  if (!dragging) return;
  const dx = x - dragStartX;
  const dy = y - dragStartY;
  // Full-canvas-width drag = one revolution; full-canvas-height drag = full
  // elevation sweep. Empirically a comfortable sensitivity.
  const A = manifest.azim_count, E = manifest.elev_count;
  const azPerPx = A / Math.max(1, canvas.width);
  const elPerPx = E / Math.max(1, canvas.height);
  const newA = ((dragStartA + Math.round(dx * azPerPx)) % A + A) % A;
  // Drag DOWN -> camera rises (trackball feel: pulling the top of the object
  // toward you reveals its top face).
  const newE = Math.max(0, Math.min(E - 1, dragStartE + Math.round(dy * elPerPx)));
  if (newA !== curA || newE !== curE) {
    curA = newA; curE = newE;
    draw();
  }
}

stage.addEventListener('pointerdown', (ev) => {
  pointerId = ev.pointerId;
  stage.setPointerCapture(pointerId);
  beginDrag(ev.clientX, ev.clientY);
  ev.preventDefault();
});
stage.addEventListener('pointermove', (ev) => {
  if (ev.pointerId !== pointerId) return;
  moveDrag(ev.clientX, ev.clientY);
});
stage.addEventListener('pointerup', (ev) => {
  if (ev.pointerId === pointerId) { pointerId = null; endDrag(); }
});
stage.addEventListener('pointercancel', endDrag);

stage.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const R = manifest.radius_count;
  if (R <= 1) return;
  // Scroll up = zoom in = smaller radius. deltaY > 0 means content scrolls
  // downward, conventionally "zoom out" -> larger radius.
  const dir = Math.sign(ev.deltaY);
  const newR = Math.max(0, Math.min(R - 1, curR + dir));
  if (newR !== curR) { curR = newR; draw(); }
}, { passive: false });

window.addEventListener('keydown', (ev) => {
  if (!manifest) return;
  if (ev.key === 'r' || ev.key === 'R') {
    curA = 0;
    curE = Math.floor(manifest.elev_count / 2);
    curR = Math.floor(manifest.radius_count / 2);
    draw();
  }
});

// -------------------- boot --------------------
(async () => {
  try {
    const resp = await fetch('manifest.json');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    manifest = await resp.json();
  } catch (err) {
    setStatus('failed to load manifest.json: ' + err.message +
              ' (serve this directory via e.g. `python3 -m http.server`)');
    return;
  }
  canvas.width  = manifest.width;
  canvas.height = manifest.height;
  curA = 0;
  curE = Math.floor(manifest.elev_count / 2);
  curR = Math.floor(manifest.radius_count / 2);
  setStatus('ready');
  await draw();
  prefetchNeighbors();
})();
</script>
</body>
</html>
)VIEWER";

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 1;

    voxr::VoxelGrid grid;
    if (!voxr::load_voxel_grid(a.voxels_path, grid)) return 1;

    voxr::Vec3 bmin = grid.origin;
    voxr::Vec3 bmax = grid.max_corner();
    voxr::Vec3 center{0.5f * (bmin.x + bmax.x),
                      0.5f * (bmin.y + bmax.y),
                      0.5f * (bmin.z + bmax.z)};
    if (!a.target_set) {
        a.tx = center.x; a.ty = center.y; a.tz = center.z;
    }
    float half_ext = std::max({0.5f * (bmax.x - bmin.x),
                               0.5f * (bmax.y - bmin.y),
                               0.5f * (bmax.z - bmin.z)});
    if (half_ext <= 0.f) half_ext = 1.f;
    if (a.radius_min <= 0.f) a.radius_min = 2.0f * half_ext;
    if (a.radius_max <= 0.f) a.radius_max = 4.0f * half_ext;

    std::error_code ec;
    fs::create_directories(a.out_dir, ec);
    if (ec) {
        std::cerr << "bake_views: cannot create '" << a.out_dir
                  << "': " << ec.message() << "\n"; return 1;
    }

    std::vector<float> radii(a.radius_count);
    if (a.radius_count == 1) {
        radii[0] = 0.5f * (a.radius_min + a.radius_max);
    } else {
        for (int r = 0; r < a.radius_count; ++r) {
            float t = static_cast<float>(r) /
                      static_cast<float>(a.radius_count - 1);
            radii[r] = a.radius_min + t * (a.radius_max - a.radius_min);
        }
    }

    voxr::RenderOptions ropts;
    ropts.shading = !a.no_shading;

    const float two_pi  = 6.28318530718f;
    const float deg2rad = 3.14159265358979f / 180.f;

    const int total = a.azim_count * a.elev_count * a.radius_count;
    std::cout << "Baking " << total << " views ("
              << a.azim_count << " azim x "
              << a.elev_count << " elev x "
              << a.radius_count << " radii) at "
              << a.width << "x" << a.height << "..." << std::endl;

    int done = 0;
    for (int ai = 0; ai < a.azim_count; ++ai) {
        float phi = (static_cast<float>(ai) / a.azim_count) * two_pi;
        for (int ei = 0; ei < a.elev_count; ++ei) {
            float elev_deg = (a.elev_count == 1)
                ? 0.5f * (a.elev_min_deg + a.elev_max_deg)
                : a.elev_min_deg +
                  (static_cast<float>(ei) / (a.elev_count - 1)) *
                  (a.elev_max_deg - a.elev_min_deg);
            float elev = elev_deg * deg2rad;
            float cos_e = std::cos(elev), sin_e = std::sin(elev);
            for (int ri = 0; ri < a.radius_count; ++ri) {
                float R   = radii[ri];
                float h   = R * sin_e;
                float rho = R * cos_e;
                voxr::Vec3 eye{a.tx + rho * std::cos(phi),
                               a.ty + h,
                               a.tz + rho * std::sin(phi)};
                voxr::Camera cam = voxr::Camera::from_look_at(
                    a.width, a.height, a.fov_y,
                    eye, {a.tx, a.ty, a.tz}, {a.ux, a.uy, a.uz});
                voxr::ImageU8 img;
                if (!voxr::render_cpu(grid, cam, img, ropts)) return 1;

                int idx = (ai * a.elev_count + ei) * a.radius_count + ri;
                char buf[64];
                std::snprintf(buf, sizeof(buf), "frame_%06d.ppm", idx);
                if (!voxr::save_ppm((fs::path(a.out_dir) / buf).string(), img))
                    return 1;

                ++done;
                if (done % 16 == 0 || done == total) {
                    std::cout << "  " << done << " / " << total << "\r"
                              << std::flush;
                }
            }
        }
    }
    std::cout << "\n";

    // manifest.json
    {
        std::ofstream f((fs::path(a.out_dir) / "manifest.json").string());
        if (!f) {
            std::cerr << "bake_views: cannot write manifest.json\n";
            return 1;
        }
        f << "{\n";
        f << "  \"width\": "        << a.width  << ",\n";
        f << "  \"height\": "       << a.height << ",\n";
        f << "  \"azim_count\": "   << a.azim_count   << ",\n";
        f << "  \"elev_count\": "   << a.elev_count   << ",\n";
        f << "  \"radius_count\": " << a.radius_count << ",\n";
        f << "  \"elev_min_deg\": " << a.elev_min_deg << ",\n";
        f << "  \"elev_max_deg\": " << a.elev_max_deg << ",\n";
        f << "  \"radii\": [";
        for (int r = 0; r < a.radius_count; ++r) {
            if (r) f << ", ";
            f << radii[r];
        }
        f << "],\n";
        f << "  \"target\": [" << a.tx << ", " << a.ty << ", " << a.tz << "],\n";
        f << "  \"up\": ["     << a.ux << ", " << a.uy << ", " << a.uz << "],\n";
        f << "  \"fov_y\": "   << a.fov_y << "\n";
        f << "}\n";
    }

    // viewer.html
    {
        std::ofstream f((fs::path(a.out_dir) / "viewer.html").string());
        if (!f) {
            std::cerr << "bake_views: cannot write viewer.html\n";
            return 1;
        }
        f << kViewerHTML;
    }

    std::cout << "Wrote " << total << " frames + manifest.json + viewer.html to "
              << a.out_dir << "\n";
    std::cout << "To view:\n";
    std::cout << "  cd " << a.out_dir << " && python3 -m http.server 8000\n";
    std::cout << "Then open: http://localhost:8000/viewer.html\n";
    return 0;
}
