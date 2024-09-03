cmake_minimum_required(VERSION 3.21)
project(Hetero-Mark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

option(HMARK_BUILD_HIP "Build all Hetero-mark benchmarks for HIP" OFF)
option(HMARK_BUILD_CUDA "Build all Hetero-mark benchmarks for CUDA" OFF)
option(HMARK_BUILD_OPENCL20 "Build all Hetero-mark benchmarks for OpenCL2.0" OFF)
option(HMARK_BUILD_OPENCL12 "Build all Hetero-mark benchmarks for OpenCL1.2" OFF)

if (${HMARK_BUILD_HIP})
    enable_language(HIP)
    find_package(HIP REQUIRED)
endif ()

if (${HMARK_BUILD_OPENCL20} OR ${HMARK_BUILD_OPENCL12})
    find_package(OpenCL REQUIRED)
endif ()

if (${HMARK_BUILD_CUDA})
    enable_language(CUDA)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(ExternalProject)
include(StyleCheck)

set(FOLDER_NAME Hetero-Mark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Include from path root
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
# Test
option(ENABLE_TEST "Compile all Hetero-Mark unit tests" OFF)
if (${ENABLE_TEST})
    enable_testing()
    ExternalProject_Add(googletest
            URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
            PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
            INSTALL_COMMAND ""
    )
    ExternalProject_Get_Property(googletest binary_dir)
    ExternalProject_Get_Property(googletest source_dir)
    add_library(gtest UNKNOWN IMPORTED)
    add_library(gtest_main UNKNOWN IMPORTED)
    set_target_properties(gtest PROPERTIES
            IMPORTED_LOCATION ${binary_dir}/libgtest.a)
    set_target_properties(gtest_main PROPERTIES
            IMPORTED_LOCATION ${binary_dir}/libgtest_main.a)
    add_dependencies(gtest googletest)
    add_dependencies(gtest_main googletest)
    include_directories(${source_dir}/include)
endif ()

add_subdirectory(src)

