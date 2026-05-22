// Minimal image types and Netpbm I/O.
//
// Color images are PPM (P6, 8-bit RGB), masks/grayscale are PGM (P5, 8-bit).
// We deliberately avoid external image libraries so the project builds with
// only a C++17 compiler.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace voxr {

struct ImageU8 {
    int width{0};
    int height{0};
    int channels{0};            // 1 or 3
    std::vector<std::uint8_t> data;  // row-major, channels interleaved

    ImageU8() = default;
    ImageU8(int w, int h, int c)
        : width(w), height(h), channels(c),
          data(static_cast<std::size_t>(w) * h * c, 0) {}

    std::size_t pixel_index(int x, int y) const {
        return (static_cast<std::size_t>(y) * width + x) * channels;
    }

    bool in_bounds(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }
};

// Returns true on success. Format is inferred from `image.channels`.
bool save_ppm(const std::string& path, const ImageU8& image);
bool save_pgm(const std::string& path, const ImageU8& image);

// Loads a binary Netpbm file (P5 or P6). Returns true on success and sets
// `out` accordingly. Comments inside the header are ignored.
bool load_pnm(const std::string& path, ImageU8& out);

}  // namespace voxr
