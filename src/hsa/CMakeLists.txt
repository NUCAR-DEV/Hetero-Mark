set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

set( FOLDER_NAME hsa )
set( SUBDIRECTORIES 
  aes_hsa
  fir_hsa
  hmm_hsa
  hist_hsa
  iir_cl12_hsa
  kmeans_hsa
  page_rank_hsa
  sw_hsa
)

###############################################################

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
