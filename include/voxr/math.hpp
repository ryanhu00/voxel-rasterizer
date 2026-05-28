// Vec3 / Mat3 utilities.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>

namespace voxr {

struct Vec3 {
    float x{0.f}, y{0.f}, z{0.f};

    constexpr Vec3() = default;
    constexpr Vec3(float v) : x(v), y(v), z(v) {}
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float& operator[](std::size_t i) { return (&x)[i]; }
    const float& operator[](std::size_t i) const { return (&x)[i]; }
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a)          { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator*(Vec3 a, Vec3 b)  { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }

inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }
inline Vec3& operator*=(Vec3& a, float s){ a = a * s; return a; }

inline float dot(Vec3 a, Vec3 b)  { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(Vec3 a)        { return std::sqrt(dot(a, a)); }
inline float length2(Vec3 a)       { return dot(a, a); }
inline Vec3 normalize(Vec3 a) {
    float n = length(a);
    return n > 0.f ? a / n : Vec3{0.f, 0.f, 0.f};
}
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline std::ostream& operator<<(std::ostream& os, Vec3 v) {
    os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
    return os;
}

// 3x3 row-major.
struct Mat3 {
    std::array<float, 9> m{};

    static Mat3 identity() {
        Mat3 r{};
        r.m[0] = r.m[4] = r.m[8] = 1.f;
        return r;
    }

    float& operator()(int r, int c)             { return m[r * 3 + c]; }
    const float& operator()(int r, int c) const { return m[r * 3 + c]; }
};

inline Vec3 operator*(const Mat3& A, Vec3 v) {
    return {A(0, 0) * v.x + A(0, 1) * v.y + A(0, 2) * v.z,
            A(1, 0) * v.x + A(1, 1) * v.y + A(1, 2) * v.z,
            A(2, 0) * v.x + A(2, 1) * v.y + A(2, 2) * v.z};
}

inline Mat3 operator*(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += A(r, k) * B(k, c);
            C(r, c) = s;
        }
    }
    return C;
}

inline Mat3 transpose(const Mat3& A) {
    Mat3 T{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            T(r, c) = A(c, r);
    return T;
}

inline Mat3 rotation_x(float a) {
    Mat3 r = Mat3::identity();
    float c = std::cos(a), s = std::sin(a);
    r(1, 1) =  c; r(1, 2) = -s;
    r(2, 1) =  s; r(2, 2) =  c;
    return r;
}
inline Mat3 rotation_y(float a) {
    Mat3 r = Mat3::identity();
    float c = std::cos(a), s = std::sin(a);
    r(0, 0) =  c; r(0, 2) =  s;
    r(2, 0) = -s; r(2, 2) =  c;
    return r;
}
inline Mat3 rotation_z(float a) {
    Mat3 r = Mat3::identity();
    float c = std::cos(a), s = std::sin(a);
    r(0, 0) =  c; r(0, 1) = -s;
    r(1, 0) =  s; r(1, 1) =  c;
    return r;
}

// OpenCV-style: x_cam = R * (X_world - eye). Camera frame is +X right,
// +Y down, +Z forward. Rows of R are right/down/forward in world coords.
inline Mat3 look_at_rotation(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = normalize(target - eye);            // forward
    Vec3 r = normalize(cross(f, normalize(up))); // right  = f x up
    Vec3 d = cross(f, r);                        // down   = f x right
    Mat3 R{};
    R(0, 0) = r.x; R(0, 1) = r.y; R(0, 2) = r.z;
    R(1, 0) = d.x; R(1, 1) = d.y; R(1, 2) = d.z;
    R(2, 0) = f.x; R(2, 1) = f.y; R(2, 2) = f.z;
    return R;
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace voxr
