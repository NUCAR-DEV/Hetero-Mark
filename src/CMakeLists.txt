set( FOLDER_NAME src )
set( SUBDIRECTORIES 
	aes_cl12
	aes_cl20 
	hmm_cl12
	hmm_cl20 
	iir_cl12
	iir_cl20 
	sw_cl12
	sw_cl20
	pagerank_cl12
	pagerank_cl20
	kerneloverhead_cl12
	kerneloverhead_cl20
	memcopy_cl12
	memcopy_cl20
	maxmem_cl12
	maxmem_cl20
	)

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
