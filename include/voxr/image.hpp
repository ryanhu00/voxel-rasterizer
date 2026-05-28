// Image type and Netpbm I/O.
// GPU plan: upload into a cudaArray + cudaTextureObject_t per view; replaces
// the CPU bilinear samplers with a single tex2D() call.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace voxr {

struct ImageU8 {
    int width{0};
    int height{0};
    int channels{0};            // 1 or 3
    std::vector<std::uint8_t> data;

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

bool save_ppm(const std::string& path, const ImageU8& image);
bool save_pgm(const std::string& path, const ImageU8& image);
bool load_pnm(const std::string& path, ImageU8& out);

}  // namespace voxr
