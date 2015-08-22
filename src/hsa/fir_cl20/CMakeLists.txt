set( FOLDER_NAME fir_cl20 )
set( SUBDIRECTORIES )
set( SRC_FILES 
  fir.cc 
)
set( HEADER_FILES )
set( TEST_FILES )
file(COPY kernels.cl DESTINATION .)

# Check is snack is available
find_file(have_snack NAMES snack.sh DOC "snack.sh file") 

# Style check
set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SRC_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Unit test
if (${TEST_FILES})
  add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
  add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
endif (${TEST_FILES})

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

if (have_snack)
  add_custom_command(
    OUTPUT kernels.o kernels.h
    COMMAND snack.sh -c kernels.cl
    DEPENDS kernels.cl
    COMMENT "Using snack to generate kernel library"
    WORKING_DIRECTORY ""
    VERBATIM
  )
  add_executable(fir_hsa_cl20 kernels.h kernels.o ${HEADER_FILES} ${SRC_FILES})
else (have_snack)
  message(WARNING "Skipping fir_hsa_cl20, snack.sh is not found")
endif (have_snack)

