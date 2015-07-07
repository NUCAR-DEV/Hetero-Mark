cmake_minimum_required (VERSION 2.6)

set (HSA_RUNTIME_PATH "$ENV{HSA_RUNTIME_PATH}")
set (M2S_ROOT "$ENV{M2S_ROOT}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_GLAGS} -m32 -std=c++0x")

link_directories(${HSA_RUNTIME_PATH}/lib)

add_executable(FIR12 main.cc FirFilter.cc)
include_directories(${HSA_RUNTIME_PATH}/include)
target_link_libraries(FIR12 HsaHelper m2s-hsa)


