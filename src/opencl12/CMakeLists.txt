include(../../CppLint.cmake)
set( FOLDER_NAME opencl12 )
set( SUBDIRECTORIES 
    aes_cl12
    fir_cl12
    hmm_cl12
    iir_cl12
    kmeans_cl12
    pagerank_cl12
    sw_cl12
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
