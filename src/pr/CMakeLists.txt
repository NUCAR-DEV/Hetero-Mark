set( FOLDER_NAME pr )
set( SUBDIRECTORIES 
  cl12
  hsa
)
set( SRC_FILES
  pr_benchmark.cc
  pr_command_line_options.cc
)
set( HEADER_FILES
  pr_benchmark.h
  pr_command_line_options.h
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

###############################################################

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library( pr ${SRC_FILES} ${HEADER_FILES} )

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
