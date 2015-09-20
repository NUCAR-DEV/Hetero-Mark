set( FOLDER_NAME runtime_helper )
set( SRC_FILES )
set( HEADER_FILES 
  runtime_helper.h
)
set( SUBDIRECTORIES 
  hsa_runtime_helper
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SRC_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
set_target_properties(${FOLDER_NAME} PROPERTIES LINKER_LANGUAGE CXX)
