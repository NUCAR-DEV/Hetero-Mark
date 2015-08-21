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
add_executable(BenchmarkRunner_unittest ${TEST_FILES} ${SOURCE_FILES} ${HEADER_FILES})
add_test(BenchmarkTest BenchmarkRunner_unittest)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )
