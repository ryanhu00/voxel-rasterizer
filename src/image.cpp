#include "voxr/image.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace voxr {

static bool save_pnm_impl(const std::string& path, const ImageU8& image,
                          const char* magic) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "save_pnm: cannot open " << path << " for writing\n";
        return false;
    }
    f << magic << "\n" << image.width << " " << image.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(image.data.data()),
            static_cast<std::streamsize>(image.data.size()));
    return static_cast<bool>(f);
}

bool save_ppm(const std::string& path, const ImageU8& image) {
    if (image.channels != 3) {
        std::cerr << "save_ppm: expected 3 channels, got " << image.channels
                  << "\n";
        return false;
    }
    return save_pnm_impl(path, image, "P6");
}

bool save_pgm(const std::string& path, const ImageU8& image) {
    if (image.channels != 1) {
        std::cerr << "save_pgm: expected 1 channel, got " << image.channels
                  << "\n";
        return false;
    }
    return save_pnm_impl(path, image, "P5");
}

// Read the next whitespace-separated token from a PNM header, skipping
// '#'-prefixed comment lines per the spec.
static bool read_token(std::ifstream& f, std::string& tok) {
    tok.clear();
    int c;
    while (true) {
        c = f.get();
        if (c == EOF) return false;
        if (c == '#') {                  // comment: skip to end of line
            while (c != EOF && c != '\n') c = f.get();
            continue;
        }
        if (!std::isspace(c)) break;
    }
    do {
        tok.push_back(static_cast<char>(c));
        c = f.get();
    } while (c != EOF && !std::isspace(c) && c != '#');
    if (c == '#') f.unget();
    return true;
}

bool load_pnm(const std::string& path, ImageU8& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "load_pnm: cannot open " << path << "\n";
        return false;
    }
    std::string magic;
    if (!read_token(f, magic)) return false;
    int channels = 0;
    if (magic == "P5")      channels = 1;
    else if (magic == "P6") channels = 3;
    else {
        std::cerr << "load_pnm: unsupported magic '" << magic << "'\n";
        return false;
    }
    std::string w_s, h_s, max_s;
    if (!read_token(f, w_s) || !read_token(f, h_s) || !read_token(f, max_s)) {
        std::cerr << "load_pnm: malformed header in " << path << "\n";
        return false;
    }
    int w     = std::stoi(w_s);
    int h     = std::stoi(h_s);
    int maxv  = std::stoi(max_s);
    if (maxv != 255) {
        std::cerr << "load_pnm: only 8-bit PNM supported (maxval=" << maxv
                  << ")\n";
        return false;
    }
    // The single whitespace byte that separates the maxval token from the
    // pixel data was already consumed by `read_token` above.
    out = ImageU8(w, h, channels);
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(out.data.size()));
    if (f.gcount() != static_cast<std::streamsize>(out.data.size())) {
        std::cerr << "load_pnm: short read in " << path << "\n";
        return false;
    }
    return true;
}

}  // namespace voxr
