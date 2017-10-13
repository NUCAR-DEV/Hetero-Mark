set( FOLDER_NAME ep )
set( SUBDIRECTORIES 
  cl12
  hc
  cuda
  hip
)
set( SRC_FILES
  ep_benchmark.cc
  ep_command_line_options.cc
)
set( HEADER_FILES
  ep_benchmark.h
  ep_command_line_options.h
)

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
  message("OpenCV is not found, skipping the EP benchmark")
endif(OpenCV_FOUND)
