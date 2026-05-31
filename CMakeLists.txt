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
    # Scope to C++ so they don't leak to nvcc, which rejects -Wpedantic etc.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-ffast-math>)
endif()

# ---- Optional CUDA -----------------------------------------------------------
# Week 2: GPU ports of the SfS and ray-march kernels. The build degrades
# gracefully to CPU-only when no CUDA toolkit is present.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(VOXR_HAVE_CUDA TRUE)
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER_VERSION}) -- GPU paths enabled")
else()
    message(STATUS "CUDA not found -- building CPU-only")
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

# ---- CUDA library ------------------------------------------------------------
# GPU kernels validated against the CPU reference in voxr_core. Targets that
# want GPU paths link voxr_cuda and get the VOXR_WITH_CUDA define.
if(VOXR_HAVE_CUDA)
    add_library(voxr_cuda STATIC
        src/reconstruct_cuda.cu
        src/render_cuda.cu
    )
    target_include_directories(voxr_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(voxr_cuda PUBLIC voxr_core)
    target_compile_definitions(voxr_cuda PUBLIC VOXR_WITH_CUDA)
endif()

# Link the GPU lib into a target when available.
function(voxr_maybe_link_cuda target)
    if(VOXR_HAVE_CUDA)
        target_link_libraries(${target} PRIVATE voxr_cuda)
    endif()
endfunction()

# ---- Applications ------------------------------------------------------------
add_executable(synth_dataset apps/synth_dataset.cpp)
target_link_libraries(synth_dataset PRIVATE voxr_core)

add_executable(reconstruct apps/reconstruct.cpp)
target_link_libraries(reconstruct PRIVATE voxr_core)
voxr_maybe_link_cuda(reconstruct)

add_executable(render apps/render.cpp)
target_link_libraries(render PRIVATE voxr_core)
voxr_maybe_link_cuda(render)

add_executable(bake_views apps/bake_views.cpp)
target_link_libraries(bake_views PRIVATE voxr_core)
voxr_maybe_link_cuda(bake_views)

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

# GPU-vs-CPU parity: the CPU reference is the oracle for both kernels.
if(VOXR_HAVE_CUDA)
    add_executable(test_cuda_parity tests/test_cuda_parity.cpp)
    target_link_libraries(test_cuda_parity PRIVATE voxr_core voxr_cuda)
    add_test(NAME test_cuda_parity COMMAND test_cuda_parity
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
