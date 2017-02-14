cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME aes_hsa)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  aes_hsa_benchmark.cc
)
set(HEADER_FILES 
  aes_hsa_benchmark.h)
set(TEST_FILES)
set(EXTRA_FILES 
  kernels.cl
)

# Unit test
if (${TEST_FILES})
  add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
  add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
endif (${TEST_FILES})

if (NOT COMPILE_HCC) 
if (have_snack AND HSA_RUNTIME)
  foreach( extra_file ${EXTRA_FILES} )
    add_custom_command(
      OUTPUT ${extra_file}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${CMAKE_CURRENT_BINARY_DIR}
    )
  endforeach( extra_file )
  add_custom_command(
    OUTPUT kernels.o kernels.h
    COMMAND ${SNACK} -c kernels.cl
    DEPENDS kernels.cl
    COMMENT "Using snack to generate kernel library"
    WORKING_DIRECTORY ""
    VERBATIM
  )
  add_executable(${FOLDER_NAME} kernels.h kernels.o ${HEADER_FILES} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME}
      aes time_measurement command_line_option benchmark ${HSA_RUNTIME})
endif (have_snack AND HSA_RUNTIME)
endif (NOT COMPILE_HCC)

