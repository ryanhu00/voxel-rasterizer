cmake_minimum_required(VERSION 3.16)
project(voxel_rasterizer
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated multi-view voxel reconstruction and rasterization"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:Release>:-ffast-math>)
endif()

# ---- Core CPU library --------------------------------------------------------
# Week 1: pure CPU. Week 2+: CUDA sources can be added to this target (or a
# sibling `voxr_cuda` target) without changing dependents.
add_library(voxr_core STATIC
    src/image.cpp
    src/camera.cpp
    src/voxel_grid.cpp
    src/scene.cpp
    src/reconstruct_cpu.cpp
    src/render_cpu.cpp
)
target_include_directories(voxr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(voxr_core PUBLIC cxx_std_17)

# ---- Applications ------------------------------------------------------------
add_executable(synth_dataset apps/synth_dataset.cpp)
target_link_libraries(synth_dataset PRIVATE voxr_core)

add_executable(reconstruct apps/reconstruct.cpp)
target_link_libraries(reconstruct PRIVATE voxr_core)

add_executable(render apps/render.cpp)
target_link_libraries(render PRIVATE voxr_core)

add_executable(bake_views apps/bake_views.cpp)
target_link_libraries(bake_views PRIVATE voxr_core)

add_executable(make_orbit_cameras apps/make_orbit_cameras.cpp)
target_link_libraries(make_orbit_cameras PRIVATE voxr_core)

# ---- Tests -------------------------------------------------------------------
enable_testing()

add_executable(test_camera tests/test_camera.cpp)
target_link_libraries(test_camera PRIVATE voxr_core)
add_test(NAME test_camera COMMAND test_camera)

add_executable(test_voxel_grid tests/test_voxel_grid.cpp)
target_link_libraries(test_voxel_grid PRIVATE voxr_core)
add_test(NAME test_voxel_grid COMMAND test_voxel_grid)

add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE voxr_core)
add_test(NAME test_pipeline COMMAND test_pipeline
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
