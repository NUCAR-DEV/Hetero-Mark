include(../../CppLint.cmake)
set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

set( FOLDER_NAME hsa )
set( SUBDIRECTORIES 
  FiniteImpulseResponseHSA
  HiddenMarkovModelHSA
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src )
add_style_check_target( ${check_target} "${check_target_src}" 
    "${subdirectories}" )

###############################################################

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
