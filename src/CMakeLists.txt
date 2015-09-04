set( FOLDER_NAME src )
set( SUBDIRECTORIES
  # Applications
        # Advanced Encryption Standard
	aes_cl12
	aes_cl20
	# Finite Impulse Filte
	fir_cl12
	fir_cl20
	# Hidden Markov Model
	hmm_cl12
	hmm_cl20
	# Infinite Impulse Filter
	iir_cl12
	iir_cl20
	# K-Means
	kmeans_cl12
	kmeans_cl20
	# Page Rank
	pagerank_cl12
	pagerank_cl20
	# Shallow Water
	sw_cl12
	sw_cl20
   # Microbenchmark
        micro-benchmark/kerneloverhead_cl20
        micro-benchmark/svmCPUcaching
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
