set( FOLDER_NAME common )
set( SUBDIRECTORIES 
  clUtil
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src )
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
