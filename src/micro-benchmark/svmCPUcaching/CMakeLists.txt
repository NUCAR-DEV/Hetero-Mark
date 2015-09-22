set( BENCHMARK_NAME svmCaching )
set( SOURCE_FILES main.cc svmcpucache.cc )
set( EXTRA_FILES kernel.cl )

############################################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${BENCHMARK_NAME} )

if(CMAKE_BUILD_TYPE MATCHES "[Tt][Bb][Bb]")
  return( )
endif()

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
	message( FATAL_ERROR "Could not locate OpenCL include & libs" )
endif( )

############################################################################

set( COMPILER_FLAGS "-std=c++11" )
set( LINKER_FLAGS " " )
set( ADDITIONAL_LIBRARIES "" )

# clUtil includes
set( CLUTIL_INCLUDE_DIRS "../../include/clUtil" )

file(GLOB INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h" )
include_directories( ${OPENCL_INCLUDE_DIRS} ${CLUTIL_INCLUDE_DIRS} )

add_executable( ${BENCHMARK_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} ${EXTRA_FILES})

# gcc/g++ specific compile options
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set( COMPILER_FLAGS "${COMPILER_FLAGS} -msse2 " )
    
    # Note: "rt" is not present on mingw
    if( UNIX )
		if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
			set( COMPILER_FLAGS "${COMPILER_FLAGS} -g" )
		endif( )
        set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} "rt" )
    endif( )
    
    if( BITNESS EQUAL 32 )
        set( COMPILER_FLAGS "${COMPILER_FLAGS} -m32 " )
        set( LINKER_FLAGS "${LINKER_FLAGS} -m32 " )
    else( )
        set( COMPILER_FLAGS "${COMPILER_FLAGS} -m64 " )
        set( LINKER_FLAGS "${LINKER_FLAGS} -m64 " )
    endif( )
    
    set( COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_GXX} " )
    set( LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_GXX} " )
    set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_GXX} )
endif( )

set_target_properties( ${BENCHMARK_NAME} PROPERTIES
                        COMPILE_FLAGS ${COMPILER_FLAGS}
                        LINK_FLAGS ${LINKER_FLAGS}
                     )
target_link_libraries( ${BENCHMARK_NAME} cl_util time_measurement command_line_option benchmark ${OPENCL_LIBRARIES} ${ADDITIONAL_LIBRARIES} ${CLUTIL_LIBRARY})

# Set output directory to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BITNESS_SUFFIX}/${CMAKE_BUILD_TYPE})

# Copy extra files to binary directory
foreach( extra_file ${EXTRA_FILES} )
    add_custom_command(
        TARGET ${BENCHMARK_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ./
        )
endforeach( extra_file )

# Group sample based on FOLDER_GROUP defined in parent folder
if( FOLDER_GROUP )
    set_target_properties(${BENCHMARK_NAME} PROPERTIES FOLDER ${FOLDER_GROUP})
endif( )

