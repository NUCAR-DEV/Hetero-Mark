# Auto-select bitness based on platform
if( NOT BITNESS )
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITNESS 64)
  else()
    set(BITNESS 32)
  endif()
endif()

# Unset OPENCL_LIBRARIES, so that corresponding arch specific libs are found when bitness is changed
unset(OPENCL_LIBRARIES CACHE)
unset(OPENCL_INCLUDE_DIRS CACHE)

if( BITNESS EQUAL 64 )
  set(BITNESS_SUFFIX x86_64)
elseif( BITNESS EQUAL 32 )
  set(BITNESS_SUFFIX x86)
else()
  message( FATAL_ERROR "Bitness specified is invalid" )
endif()

# Set CMAKE_BUILD_TYPE (default = Release)
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release)
endif()

############################################################################

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
	message( "OpenCL include file and libraries not found. OpenCL benchmarks \
      will be skipped." )
endif( )
