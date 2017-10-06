set( FOLDER_NAME knn )
set( SUBDIRECTORIES 
  cuda
  cuda_ws
  cuda_ws_v2
  )
set( SRC_FILES
  knn_benchmark.cc
  knn_command_line_options.cc
)
set( HEADER_FILES
  knn_benchmark.h
  knn_command_line_options.h
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library( knn ${SRC_FILES} ${HEADER_FILES} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
