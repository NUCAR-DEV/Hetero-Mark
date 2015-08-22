set( FOLDER_NAME Timer )
set( SUBDIRECTORIES )
set( SRC_FILES
  TimeKeeperImpl.cc
  TimeKeeperSummaryPrinter.cc
  TimeMeasurementImpl.cc
  TimerImpl.cc)
set( HEADER_FILES 
  TimeKeeper.h
  TimeKeeperImpl.h
  TimeKeeperSummaryPrinter.h
  TimeMeasurement.h
  TimeMeasurementImpl.h
  Timer.h
  TimerImpl.h)
set( TEST_FILES
  TimeKeeperImpl_unittest.cc)

# Style check
set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SRC_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Unit test
add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
add_test(${FOLDER_NAME} ${FOLDER_NAME}_unittest)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library(Timer ${SRC_FILES} ${HEADER_FILES})
