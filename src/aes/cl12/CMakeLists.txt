set( BENCHMARK_NAME aes_cl12 )
set( SOURCE_FILES aes_cl12_benchmark.cc main.cc )
set( HEADER_FILES aes_cl12_benchmark.h )
set( EXTRA_FILES kernels.cl )

############################################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${BENCHMARK_NAME} )

if( OPENCL_INCLUDE_DIRS AND OPENCL_LIBRARIES )
  add_executable( ${BENCHMARK_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} ${EXTRA_FILES})

  target_link_libraries( ${BENCHMARK_NAME} 
      aes cl_util time_measurement command_line_option benchmark 
      ${OPENCL_LIBRARIES})

  # Copy extra files to binary directory
  foreach( extra_file ${EXTRA_FILES} )
    add_custom_command(
        TARGET ${BENCHMARK_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${CMAKE_CURRENT_BINARY_DIR}
        )
  endforeach( extra_file )
endif()
