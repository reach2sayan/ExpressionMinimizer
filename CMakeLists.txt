cmake_minimum_required(VERSION 3.20)

project(ExpressionMinimizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DOXYGEN_GENERATE_LATEX YES)
set(DOXYGEN_LATEX_CMD_NAME /usr/bin/pdflatex)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(EXPRMIN_ENABLE_LOGGING_DEFAULT OFF)
else ()
    set(EXPRMIN_ENABLE_LOGGING_DEFAULT ON)
endif ()
option(ENABLE_LOGGING "Enable per-iteration logger (requires Boost.Signals2 + spdlog)" ${EXPRMIN_ENABLE_LOGGING_DEFAULT})

option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)
option(ENABLE_MKL "Enable Intel MKL backend for Eigen when available" OFF)

include(CheckCXXCompilerFlag)
include(CTest)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(EXPRMIN_COMPILER_MSVC ON)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(EXPRMIN_COMPILER_GCC ON)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(EXPRMIN_COMPILER_CLANG ON)
else ()
    message(STATUS "Compiler ${CMAKE_CXX_COMPILER_ID}: using portable C++23 defaults only")
endif ()

# AVX2 — Eigen auto-enables AVX2/FMA vectorization when __AVX2__ is defined
# by the compiler (via -mavx2 / -march=x86-64-v3 / /arch:AVX2).
if (EXPRMIN_COMPILER_MSVC)
    check_cxx_compiler_flag(/arch:AVX2 HAVE_AVX2)
    if (HAVE_AVX2)
        add_compile_options(/arch:AVX2)
        message(STATUS "AVX2: enabled (/arch:AVX2)")
    else ()
        message(STATUS "AVX2: not supported by this compiler")
    endif ()
else ()
    check_cxx_compiler_flag(-mavx2 HAVE_AVX2)
    if (HAVE_AVX2)
        if (ENABLE_NATIVE_ARCH)
            add_compile_options(-march=native)
            message(STATUS "AVX2: enabled (-march=native)")
        else ()
            add_compile_options(-march=x86-64-v3) # x86-64-v3 ⊃ AVX2 + FMA + BMI2
            message(STATUS "AVX2: enabled (-march=x86-64-v3)")
        endif ()
    else ()
        message(STATUS "AVX2: not supported by this compiler")
    endif ()
endif ()


add_library(eigen_backend INTERFACE)
if (ENABLE_MKL)
    # Prefer modern oneAPI (provides MKLConfig.cmake); fall back to legacy apt install.
    find_package(MKL CONFIG QUIET)
    if (MKL_FOUND)
        message(STATUS "MKL: found (oneAPI) — enabling Eigen MKL backend")
        target_compile_definitions(eigen_backend INTERFACE EIGEN_USE_MKL_ALL)
        target_link_libraries(eigen_backend INTERFACE MKL::MKL)
    else ()
        # Legacy intel-mkl (apt): headers under /usr/include/mkl, single SDL libmkl_rt.
        find_path(MKL_INCLUDE_DIR mkl.h
                PATHS /usr/include/mkl /usr/include
                PATH_SUFFIXES mkl)
        find_library(MKL_RT_LIB mkl_rt)
        if (MKL_INCLUDE_DIR AND MKL_RT_LIB)
            message(STATUS "MKL: found (legacy apt) — enabling Eigen MKL backend")
            if (NOT TARGET MKL::MKL)
                add_library(MKL::MKL IMPORTED INTERFACE)
                set_target_properties(MKL::MKL PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                        INTERFACE_LINK_LIBRARIES "${MKL_RT_LIB}")
            endif ()
            target_compile_definitions(eigen_backend INTERFACE EIGEN_USE_MKL_ALL)
            target_link_libraries(eigen_backend INTERFACE MKL::MKL)
        else ()
            message(STATUS "MKL: not found — using default Eigen backend")
            target_compile_definitions(eigen_backend INTERFACE EIGEN_NO_MKL)
        endif ()
    endif ()
else ()
    message(STATUS "MKL: disabled by user")
    target_compile_definitions(eigen_backend INTERFACE EIGEN_NO_MKL)
endif ()

find_package(Threads REQUIRED)

# ── ExpressionDifferentiator submodule ─────────────────────────────────────
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMODULE_RESULT
    )
    if (NOT GIT_SUBMODULE_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init --recursive failed (${GIT_SUBMODULE_RESULT})")
    endif ()
endif ()
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/expression_differentiator/CMakeLists.txt")
    message(FATAL_ERROR "deps/expression_differentiator not found — run: git submodule update --init")
endif ()
if (EXPRMIN_COMPILER_MSVC)
    set(DIFF_BUILD_BENCHMARKS OFF CACHE BOOL "Build ExpressionDifferentiator benchmarks" FORCE)
    set(DIFF_BUILD_COMPARE OFF CACHE BOOL "Build ExpressionDifferentiator comparison benchmarks" FORCE)
endif ()
add_subdirectory(deps/expression_differentiator)

set(_exprmin_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)
set(BUILD_TESTING ${_exprmin_build_testing} CACHE BOOL "Build testing" FORCE)

add_library(minimizer INTERFACE)
target_include_directories(minimizer INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(minimizer INTERFACE diff Eigen3::Eigen eigen_backend Threads::Threads)
target_compile_features(minimizer INTERFACE cxx_std_23)
target_compile_options(minimizer INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

# ── Optional iteration logger (boost::signals2 + spdlog) ─────────────────────
if (ENABLE_LOGGING)
    # Boost — header-only signals2; the project already uses mp11 so Boost
    # headers should already be on the path, but make it explicit here.
    find_package(Boost QUIET)
    if (NOT Boost_FOUND)
        message(FATAL_ERROR "ENABLE_LOGGING requires Boost (Signals2 headers). "
                "Install libboost-dev or set BOOST_ROOT.")
    endif ()

    # spdlog — prefer system install, fall back to FetchContent.
    find_package(spdlog QUIET)
    if (NOT spdlog_FOUND)
        message(STATUS "spdlog: not found via find_package — fetching from GitHub")
        include(FetchContent)
        FetchContent_Declare(spdlog
                GIT_REPOSITORY https://github.com/gabime/spdlog.git
                GIT_TAG v1.14.1
                GIT_SHALLOW TRUE)
        set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(spdlog)
    else ()
        message(STATUS "spdlog: found (system, version ${spdlog_VERSION})")
    endif ()

    target_compile_definitions(minimizer INTERFACE EXPRMIN_LOGGING)
    target_link_libraries(minimizer INTERFACE Boost::headers spdlog::spdlog_header_only)
    message(STATUS "ENABLE_LOGGING: ON — iteration logger enabled")
endif ()

# ── nlohmann/json (playground only) ──────────────────────────────────────────
find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann/json: not found — fetching from GitHub")
    FetchContent_Declare(nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
            DOWNLOAD_NO_EXTRACT TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
    # Single-header download lands at nlohmann_json_SOURCE_DIR/json.hpp.
    # To support #include <nlohmann/json.hpp>, copy it into a nlohmann/ subdir.
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/nlohmann_include/nlohmann)
    file(COPY ${nlohmann_json_SOURCE_DIR}/json.hpp
         DESTINATION ${CMAKE_BINARY_DIR}/nlohmann_include/nlohmann)
    add_library(nlohmann_json_header INTERFACE)
    target_include_directories(nlohmann_json_header INTERFACE
            ${CMAKE_BINARY_DIR}/nlohmann_include)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_header)
else ()
    message(STATUS "nlohmann/json: found (system)")
endif ()

# ── Playground ───────────────────────────────────────────────────────────────
add_executable(playground src/playground.cpp)
target_link_libraries(playground PRIVATE minimizer nlohmann_json::nlohmann_json)
target_include_directories(playground PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/expression_differentiator)
set_target_properties(playground PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
if (EXPRMIN_COMPILER_GCC)
    target_compile_options(playground PRIVATE -Wall -Wextra)
elseif (EXPRMIN_COMPILER_CLANG)
    target_compile_options(playground PRIVATE -Wall -Wextra)
endif ()

option(BUILD_DOCS "Build Doxygen API documentation" OFF)
if (BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_PROJECT_NAME "ExpressionMinimizer")
    set(DOXYGEN_PROJECT_BRIEF "Header-only C++23 expression minimization library")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    doxygen_add_docs(docs
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            COMMENT "Generating API documentation with Doxygen"
    )
    message(STATUS "Doxygen: docs target enabled (output: ${CMAKE_CURRENT_BINARY_DIR}/docs)")
endif ()

if (BUILD_TESTING)
    FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(minimizer_tests tests/minimizer_tests.cpp)
    target_link_libraries(minimizer_tests PRIVATE minimizer gtest_main)
    target_include_directories(minimizer_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/expression_differentiator)
    set_target_properties(minimizer_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

    if (EXPRMIN_COMPILER_MSVC)
        target_compile_options(minimizer_tests PRIVATE
                /W4
                /permissive-
                /Zc:__cplusplus
                /external:W0
                /external:anglebrackets)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
        foreach (target gtest gtest_main gmock gmock_main)
            if (TARGET ${target})
                target_compile_options(${target} PRIVATE /W0 /wd4668)
                set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
            endif ()
        endforeach ()
    elseif (EXPRMIN_COMPILER_GCC)
        target_compile_options(minimizer_tests PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
    elseif (EXPRMIN_COMPILER_CLANG)
        target_compile_options(minimizer_tests PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
    endif ()

    include(GoogleTest)
    gtest_discover_tests(minimizer_tests DISCOVERY_MODE PRE_TEST)
endif ()
