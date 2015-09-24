set( FOLDER_NAME cl_util )
set( HEADER_FILE 
  cl_error.h
  cl_file.h
  cl_profiler.h
  cl_runtime.h
  cl_util.h
)

set( SRC_FILE
  cl_error.cc
  cl_file.cc
  cl_profiler.cc
  cl_runtime.cc
  cl_util.cc
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${HEADER_FILE})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Find OpenCL include and libs
find_path( OPENCL_INCLUDE_DIRS
    NAMES OpenCL/cl.h CL/cl.h
    HINTS ../../include/ $ENV{AMDAPPSDKROOT}/include
)
mark_as_advanced(OPENCL_INCLUDE_DIRS)

find_library( OPENCL_LIBRARIES
	NAMES OpenCL
	HINTS $ENV{AMDAPPSDKROOT}/lib
	PATH_SUFFIXES ${PLATFORM}${BITNESS} ${BITNESS_SUFFIX}
)
mark_as_advanced( OPENCL_LIBRARIES )

if( OPENCL_INCLUDE_DIRS STREQUAL "" OR OPENCL_LIBRARIES STREQUAL "")
	message( FATAL_ERROR "Could not locate OpenCL include & libs" )
endif( )


###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

include_directories( ${OPENCL_INCLUDE_DIRS} ${CLUTIL_INCLUDE_DIRS} )

add_library(${FOLDER_NAME} ${HEADER_FILE} ${SRC_FILE})
