set( FOLDER_NAME benchmark )
set( SUBDIRECTORIES )
set( SRC_FILES 
  benchmark_runner.cc)
set( HEADER_FILES 
  benchmark.h
  benchmark_runner.h)
set( TEST_FILES
    benchmark_runner_test.cc)

# Unit test
if (enable-test)
  add_executable(${FOLDER_NAME}_test 
    ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_test gtest gtest_main)
  add_test(${FOLDER_NAME}_test ${FOLDER_NAME}_test)
endif (enable-test)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
