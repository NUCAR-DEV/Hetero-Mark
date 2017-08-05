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
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )


if(OPENCL_INCLUDE_DIRS AND OPENCL_LIBRARIES)
  include_directories( ${OPENCL_INCLUDE_DIRS} ${CLUTIL_INCLUDE_DIRS} )
  add_library(${FOLDER_NAME} ${HEADER_FILE} ${SRC_FILE})
endif()
