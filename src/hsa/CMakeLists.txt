include(../../CppLint.cmake)
set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

set( FOLDER_NAME hsa )
set( SUBDIRECTORIES 
  fir_hsa
  hmm_hsa
  hist_hsa
  iir_cl12_hsa
  kmeans_hsa
  page_rank_hsa
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src )
add_style_check_target( ${check_target} "${check_target_src}" 
  "${SUBDIRECTORIES}" )

###############################################################

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
