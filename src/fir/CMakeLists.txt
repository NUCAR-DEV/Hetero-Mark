set( FOLDER_NAME fir )
set( SUBDIRECTORIES 
  cl12
  cl20
  hsa
)
set( SRC_FILES
  fir_benchmark.cc
  fir_command_line_options.cc
)
set( HEADER_FILES
  fir_benchmark.h
  fir_command_line_options.h
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library( fir ${SRC_FILES} ${HEADER_FILES} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
