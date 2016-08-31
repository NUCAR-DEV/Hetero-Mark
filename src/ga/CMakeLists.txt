set( FOLDER_NAME ga )
set( SUBDIRECTORIES 
  hc
)
set( SRC_FILES
  ga_benchmark.cc
  ga_command_line_options.cc
)
set( HEADER_FILES
  ga_benchmark.h
  ga_command_line_options.h
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library( ga ${SRC_FILES} ${HEADER_FILES} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
