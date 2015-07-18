cmake_minimum_required (VERSION 2.6)

set (HSA_RUNTIME_PATH "$ENV{HSA_RUNTIME_PATH}")
set (M2S_ROOT "$ENV{M2S_ROOT}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_GLAGS} -g -std=c++0x")

#link_directories(${M2S_ROOT}/lib/.libs/)
link_directories(${HSA_RUNTIME_PATH}/lib)

add_executable(IIR12 main.cc IirFilter.cc)
include_directories(${HSA_RUNTIME_PATH}/include)
target_link_libraries(IIR12 HsaHelper hsa-runtime64)


