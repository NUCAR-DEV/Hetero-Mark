set(FOLDER_NAME ga)
set(SUBDIRECTORIES 
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

if (${OpenCV_FOUND})

  include_directories(${OpenCV_INCLUDE_DIRS})

  add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})

  foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
  endforeach( subdir )

else ()
  message(WARNING "OpenCV is not found, skipping the BE benchmark")
endif()
