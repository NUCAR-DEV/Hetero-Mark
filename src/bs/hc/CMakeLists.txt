set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

set( FOLDER_NAME bs_hc )
set( SUBDIRECTORIES )
set( SRC_FILES 
  main.cc
  bs_hc_benchmark.cc
)
set( HEADER_FILES 
  bs_hc_benchmark.h
)
set( TEST_FILES )
set( EXTRA_FILES 
  kernels.cl
)

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

# Unit test
if (${TEST_FILES})
  add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
  add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
endif (${TEST_FILES})

###############################################################

if (have_snack AND HSA_RUNTIME)
  add_custom_target(
    ${FOLDER_NAME}_exe ALL
    DEPENDS ${FOLDER_NAME}
  )

  add_custom_command(
    OUTPUT ${FOLDER_NAME}
    COMMAND hcc `hcc-config --cxxflags --ldflags`
      -I${CMAKE_SOURCE_DIR}
      ${SRC_FILES}
      -L${CMAKE_BINARY_DIR}/src/bs/ -lbs -lm
      -L${CMAKE_BINARY_DIR}/src/common/time_measurement/ -ltime_measurement
      -L${CMAKE_BINARY_DIR}/src/common/command_line_option/ -lcommand_line_option
      -L${CMAKE_BINARY_DIR}/src/common/benchmark/ -lbenchmark
      -o ${CMAKE_CURRENT_BINARY_DIR}/${FOLDER_NAME}
    DEPENDS bs time_measurement command_line_option benchmark 
      ${HEADER_FILES} ${TEST_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif (have_snack AND HSA_RUNTIME)
