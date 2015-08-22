set( FOLDER_NAME Benchmark )
set( SUBDIRECTORIES )
set( SOURCE_FILES 
  BenchmarkRunner.cc)
set( HEADER_FILES 
  Benchmark.h
  BenchmarkRunner.h)
set( TEST_FILES
    BenchmarkRunner_unittest.cc)

# Style check
set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SOURCE_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Unit test
add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library(Benchmark ${SOURCE_FILES} ${HEADER_FILES})
