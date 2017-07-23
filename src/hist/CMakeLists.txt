set( FOLDER_NAME hist )
set( SUBDIRECTORIES 
  cl12
  cl20
  hc
  cuda
)
set( SRC_FILES
  hist_benchmark.cc
  hist_command_line_options.cc
)
set( HEADER_FILES
  hist_benchmark.h
  hist_command_line_options.h
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library( hist ${SRC_FILES} ${HEADER_FILES} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
