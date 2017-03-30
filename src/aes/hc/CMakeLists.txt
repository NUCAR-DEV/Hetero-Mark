cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME aes_hc)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  aes_hc_benchmark.cc
)
set(HEADER_FILES
  aes_hc_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (COMPILE_HCC)

  # Unit test
  if (${TEST_FILES})
    add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
    target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
    add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
  endif (${TEST_FILES})

  add_executable(${FOLDER_NAME} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME} 
    aes time_measurement command_line_option benchmark)
  set_target_properties(${FOLDER_NAME} PROPERTIES
    COMPILE_FLAGS ${COMPILER_FLAGS}
    LINK_FLAGS ${LINKER_FLAGS})

endif (COMPILE_HCC)

