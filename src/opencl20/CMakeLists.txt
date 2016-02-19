set( FOLDER_NAME opencl20 )
set( SUBDIRECTORIES 
    aes_cl20
    fir_cl20
    hmm_cl20
    iir_cl20
    kmeans_cl20
    pagerank_cl20
    sw_cl20
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

if(OPENCL_INCLUDE_DIRS AND OPENCL_LIBRARIES)
  foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
  endforeach( subdir )
endif()
