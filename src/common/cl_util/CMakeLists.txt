set( FOLDER_NAME cl_util )
set( HEADER_FILE 
  cl_benchmark.h
  cl_error.h
  cl_file.h
  cl_profiler.h
  cl_runtime.h
  cl_util.h
)

set( SRC_FILE
  cl_benchmark.cc
  cl_error.cc
  cl_file.cc
  cl_profiler.cc
  cl_runtime.cc
  cl_util.cc
)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 3.21 )
project( ${FOLDER_NAME} )


if(${HMARK_BUILD_OPENCL20} OR ${HMARK_BUILD_OPENCL12})
  add_library(${FOLDER_NAME} ${HEADER_FILE} ${SRC_FILE})
  target_include_directories(${FOLDER_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(${FOLDER_NAME} PUBLIC ${OpenCL_LIBRARIES})
endif()
