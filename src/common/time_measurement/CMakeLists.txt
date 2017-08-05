set(FOLDER_NAME time_measurement)
set(SUBDIRECTORIES)
set(SRC_FILES
  time_keeper_impl.cc
  time_keeper_summary_printer.cc
  time_measurement_impl.cc
  timer_impl.cc)
set( HEADER_FILES 
  time_keeper.h
  time_keeper_impl.h
  time_keeper_summary_printer.h
  time_measurement.h
  time_measurement_impl.h
  timer.h
  timer_impl.h)
set( TEST_FILES
  time_keeper_impl_test.cc)

# Unit test
if (enable-test)
  add_executable(${FOLDER_NAME}_test 
    ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_test gtest gtest_main)
  add_test(${FOLDER_NAME} ${FOLDER_NAME}_test)
endif (enable-test)

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
if (COMPILE_HCC)
  set_target_properties(${FOLDER_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})
endif (COMPILE_HCC)
