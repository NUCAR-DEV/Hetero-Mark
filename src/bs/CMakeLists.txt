set( FOLDER_NAME bs )
set( SUBDIRECTORIES 
  cl12
  cuda
  hip
)
set( SRC_FILES
  bs_benchmark.cc
  bs_command_line_options.cc
)
set( HEADER_FILES
  bs_benchmark.h
  bs_command_line_options.h
)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )
find_package(OpenCV)


if (${OpenCV_FOUND})
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})

  foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
  endforeach(subdir)

else ()
  message(WARNING "OpenCV is not found, skipping the BS benchmark")
endif()
###############################################################

# Group samples by folder
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

#add_library( bs ${SRC_FILES} ${HEADER_FILES} )

