cmake_minimum_required (VERSION 2.6)

set (HSA_RUNTIME_PATH "$ENV{HSA_RUNTIME_PATH}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_GLAGS} -std=c++0x")

link_directories(${HSA_RUNTIME_PATH}/lib)

add_executable(kmeans12 main.cc Kmeans.cc)
include_directories(${HSA_RUNTIME_PATH}/include)
target_link_libraries(kmeans12 HsaHelper hsa-runtime64)


