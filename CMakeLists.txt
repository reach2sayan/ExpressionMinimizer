cmake_minimum_required(VERSION 3.20)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

project(ExpressionMinimizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_NATIVE_ARCH "Build optimized for this machine" OFF)

if (MSVC)
    add_compile_options(/arch:AVX2)
else ()
    if (ENABLE_NATIVE_ARCH)
        add_compile_options(-march=native)
    else ()
        add_compile_options(-march=x86-64-v3)
    endif ()
endif ()

# ── ExpressionDifferentiator submodule ─────────────────────────────────────
# Cloned into deps/expression_differentiator via:
#   git submodule add https://github.com/reach2sayan/ExpressionSolver.git \
#       deps/expression_differentiator
add_subdirectory(deps/expression_differentiator)

# ── Eigen 3.4.0 ────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# ── GoogleTest ─────────────────────────────────────────────────────────────
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── minimizer library ──────────────────────────────────────────────────────
add_library(minimizer SHARED src/minimizer.cpp)

target_include_directories(minimizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(minimizer PUBLIC diff Eigen3::Eigen)

set_target_properties(minimizer PROPERTIES
    CXX_STANDARD              23
    CXX_STANDARD_REQUIRED     ON
    POSITION_INDEPENDENT_CODE ON
)

if (NOT MSVC)
    target_compile_options(minimizer PRIVATE -Wall -Wpedantic -Wfatal-errors -Wextra)
else ()
    target_compile_options(minimizer PRIVATE /Wall)
    set_target_properties(minimizer PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# ── tests ──────────────────────────────────────────────────────────────────
add_executable(minimizer_tests tests/minimizer_tests.cpp)
target_link_libraries(minimizer_tests PRIVATE minimizer gtest_main)
# Expose the expression_differentiator.hpp umbrella header from the submodule.
target_include_directories(minimizer_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/expression_differentiator)
set_target_properties(minimizer_tests PROPERTIES
    CXX_STANDARD          23
    CXX_STANDARD_REQUIRED ON
)

enable_testing()
add_test(NAME minimizer_tests COMMAND minimizer_tests)
