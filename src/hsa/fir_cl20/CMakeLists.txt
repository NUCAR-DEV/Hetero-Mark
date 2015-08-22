set( FOLDER_NAME fir_cl20 )
set( SUBDIRECTORIES )
set( SOURCE_FILES 
  fir.cc)
set( HEADER_FILES )
set( TEST_FILES )

# Check is snack is available
find_program(have_snack snack.sh)

# Style check
set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SOURCE_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Unit test
if (${TEST_FILES})
  add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SOURCE_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
  add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
endif (${TEST_FILES})

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

if (${have_snack})
  add_custom_target(
      NAME kernels.o
      COMMAND snack.sh -c kernels.cl
      DEPENDS kernels.cl
  )
  add_executable(fir_hsa_cl20 kernels.o ${HEADER_FILES} ${SOURCE_FILES})
else (${have_snack})
  message(WARNING "Skipping fir_hsa_cl20, snack.sh is not found")
endif (${have_snack})

