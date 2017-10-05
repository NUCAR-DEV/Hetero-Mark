set( FOLDER_NAME bs )
set( SUBDIRECTORIES 
  cl12
  hc
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


if (OpenCV_FOUND)

  include_directories("/usr/include/x86_64-linux-gnu")
  add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
  if (COMPILE_HCC)
    set_target_properties(${FOLDER_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})
  endif (COMPILE_HCC)

  foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
  endforeach(subdir)

else (OpenCV_FOUND)
  message("OpenCV is not found, skipping the BS benchmark")
  endif(OpenCV_FOUND)
###############################################################

# Group samples by folder
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

#add_library( bs ${SRC_FILES} ${HEADER_FILES} )

