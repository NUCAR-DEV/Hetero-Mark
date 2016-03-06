set( BENCHMARK_NAME kmeans_cl12)
set( SOURCE_FILES main.cc kmeans_cl12.cc)
set( HEADER_FILES include/kmeans_cl12.h )
set( EXTRA_FILES kmeans_cl12_kernel.cl )

############################################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${BENCHMARK_NAME} )

if(CMAKE_BUILD_TYPE MATCHES "[Tt][Bb][Bb]")
  return( )
endif()

set( LINKER_FLAGS " " )
set( ADDITIONAL_LIBRARIES "-lm -pthread" )

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


# Copy extra files to binary directory
foreach( extra_file ${EXTRA_FILES} )
    add_custom_command(
        TARGET ${BENCHMARK_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${CMAKE_CURRENT_BINARY_DIR}
        )
endforeach( extra_file )

# Group sample based on FOLDER_GROUP defined in parent folder
if( FOLDER_GROUP )
    set_target_properties(${BENCHMARK_NAME} PROPERTIES FOLDER ${FOLDER_GROUP})
endif( )

