cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME ga)
set(SUBDIRECTORIES 
  hc
  cuda
  hip
  cl12
)
set(SRC_FILES
  ga_benchmark.cc
  ga_command_line_options.cc
)
set(HEADER_FILES
  ga_benchmark.h
  ga_command_line_options.h
)
find_package(OpenCV)

if (OpenCV_FOUND)

  include_directories("/usr/include/x86_64-linux-gnu")

  add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
  if (COMPILE_HCC)
    set_target_properties(${FOLDER_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})
  endif (COMPILE_HCC)

  foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
  endforeach( subdir )

else (OpenCV_FOUND)
  message("OpenCV is not found, skipping the BE benchmark")
endif(OpenCV_FOUND) 
