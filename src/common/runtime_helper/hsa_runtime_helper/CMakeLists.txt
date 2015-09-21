set( FOLDER_NAME hsa_runtime_helper )
set( SRC_FILES 
  aql_queue.cc
  hsa_runtime_helper.cc
  hsa_agent.cc
)
set( HEADER_FILES 
  aql_queue.h
  hsa_runtime_helper.h
  hsa_agent.h
)
set( TEST_FILES )

set( SUBDIRECTORIES )

set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SRC_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

find_file(have_hsa_include 
  NAMES hsa.h 
  HINTS "/opt/hsa/include/"
  DOC "hsa.h file")

# Unit test
if (hsa_hsa_include AND enable-test)
  add_executable(${FOLDER_NAME}_test 
    ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_test gtest gtest_main)
  add_test(${FOLDER_NAME} ${FOLDER_NAME}_test)
endif (hsa_hsa_include AND enable-test)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

if (have_hsa_include)
  message(${have_hsa_include})
  include_directories("/opt/hsa/include/")
  add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
else (have_hsa_include)
  message(WARNING "Skipping ${FOLDER_NAME}, hsa.h not found")
endif (have_hsa_include)
