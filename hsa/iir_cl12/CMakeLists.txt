cmake_minimum_required (VERSION 2.6)

set (HSA_RUNTIME_PATH "$ENV{HSA_RUNTIME_PATH}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_GLAGS} -std=c++0x")

#find_library(hsa_runtime hsa-runtime64 HINTS "${HSA_RUNTIME_PATH}/lib")
#if (NOT hsa_runtime)
#	message(FATAL_ERROR "Library 'hsa-runtime64` not found in `${HSA_RUNTIME_PATH}/lib`")
#endif()

#find_library(hsa_runtime_ext hsa-runtime-ext64 HINTS "${HSA_RUNTIME_PATH}/lib")
#if (NOT hsa_runtime_ext)
#	message(FATAL_ERROR "Library 'hsa-runtime-ext64` not found in `${HSA_RUNTIME_PATH}/lib`")
#endif()

link_directories(${HSA_RUNTIME_PATH}/lib)

add_executable(IIR main.cc IirFilter.cc)
include_directories(${HSA_RUNTIME_PATH}/include)
target_link_libraries(IIR HsaHelper hsa-runtime64)


