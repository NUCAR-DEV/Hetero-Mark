set(BENCHMARK_NAME aes_cl12)
set(SOURCE_FILES aes_cl12_benchmark.cc main.cc)
set(HEADER_FILES aes_cl12_benchmark.h)
set(EXTRA_FILES kernels.cl)

############################################################################

if(${HMARK_BUILD_OPENCL12})
  add_executable( ${BENCHMARK_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} ${EXTRA_FILES})
  target_include_directories(${BENCHMARK_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
  target_compile_definitions(${BENCHMARK_NAME} PUBLIC -DCL_TARGET_OPENCL_VERSION=120)
  target_link_libraries( ${BENCHMARK_NAME}
          aes cl_util time_measurement command_line_option benchmark ${OpenCL_LIBRARIES} )

  # Copy extra files to binary directory
  foreach(extra_file ${EXTRA_FILES})
    add_custom_command(
        TARGET ${BENCHMARK_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${CMAKE_CURRENT_BINARY_DIR}
        )
  endforeach(extra_file)
endif()
