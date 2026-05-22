// Minimal header-only test harness. No external deps; main() returns the
// number of failed assertions so CTest reports the test as failed.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace voxr_test {

inline int& failure_count() {
    static int n = 0;
    return n;
}

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& detail = {}) {
    if (!ok) {
        ++failure_count();
        std::fprintf(stderr, "FAIL %s:%d: %s%s%s\n", file, line, expr,
                     detail.empty() ? "" : "  ",
                     detail.c_str());
    }
}

}  // namespace voxr_test

#define VOXR_EXPECT(cond)                                                      \
    voxr_test::report(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

#define VOXR_EXPECT_NEAR(a, b, tol)                                            \
    do {                                                                       \
        double _da = static_cast<double>(a);                                   \
        double _db = static_cast<double>(b);                                   \
        bool _ok = std::fabs(_da - _db) <= static_cast<double>(tol);           \
        std::string _detail;                                                   \
        if (!_ok) {                                                            \
            char _buf[256];                                                    \
            std::snprintf(_buf, sizeof(_buf), "lhs=%.6g rhs=%.6g tol=%.6g",    \
                          _da, _db, static_cast<double>(tol));                 \
            _detail = _buf;                                                    \
        }                                                                      \
        voxr_test::report(_ok, #a " ~= " #b, __FILE__, __LINE__, _detail);     \
    } while (0)

#define VOXR_TEST_RETURN()                                                     \
    do {                                                                       \
        int _n = voxr_test::failure_count();                                   \
        if (_n == 0) std::fprintf(stderr, "PASS\n");                           \
        else         std::fprintf(stderr, "%d test(s) failed.\n", _n);         \
        return _n;                                                             \
    } while (0)
