#include "test_harness.hpp"

#include "voxr/camera.hpp"
#include "voxr/math.hpp"

#include <cmath>

int main() {
    using namespace voxr;

    // ---- Projection / unprojection round-trip --------------------------------
    Camera cam = Camera::from_look_at(640, 480, 0.9f,
                                      Vec3{0.f, 0.f, 5.f},
                                      Vec3{0.f, 0.f, 0.f},
                                      Vec3{0.f, 1.f, 0.f});

    // Pick a world point in front of the camera and project to pixels.
    Vec3 P{0.4f, -0.2f, 1.f};
    float u, v, depth;
    VOXR_EXPECT(cam.project(P, u, v, depth));
    VOXR_EXPECT(u >= 0.f && u < cam.width);
    VOXR_EXPECT(v >= 0.f && v < cam.height);

    // Unproject the pixel and confirm the resulting ray passes through P.
    Vec3 ro, rd;
    cam.unproject_ray(u, v, ro, rd);
    Vec3 to_P = P - ro;
    float t = dot(to_P, rd);
    Vec3 closest = ro + rd * t;
    VOXR_EXPECT_NEAR(length(P - closest), 0.0, 1e-3);

    // ---- A world point at the principal axis maps to the principal point ----
    Vec3 P2{0.f, 0.f, 0.f};   // target
    VOXR_EXPECT(cam.project(P2, u, v, depth));
    VOXR_EXPECT_NEAR(u, cam.cx, 1e-3);
    VOXR_EXPECT_NEAR(v, cam.cy, 1e-3);
    VOXR_EXPECT(depth > 0.f);

    // ---- The world "up" direction projects above the principal point --------
    Vec3 P_up{0.f, 0.3f, 0.f};
    VOXR_EXPECT(cam.project(P_up, u, v, depth));
    VOXR_EXPECT(v < cam.cy);

    // ---- World +X projects to the right of the principal point --------------
    Vec3 P_right{0.3f, 0.f, 0.f};
    VOXR_EXPECT(cam.project(P_right, u, v, depth));
    VOXR_EXPECT(u > cam.cx);

    // ---- Behind-the-camera point fails to project ----------------------------
    Vec3 P_behind{0.f, 0.f, 10.f};   // same side as the eye => z_cam < 0
    VOXR_EXPECT(!cam.project(P_behind, u, v, depth));

    VOXR_TEST_RETURN();
}
