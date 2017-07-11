cmake_minimum_required(VERSION 3.2)
set(FOLDER_NAME kmeans)
set(SUBDIRECTORIES 
  cl12
  cl20
  hc
  hsa
)
set(SRC_FILES
  kmeans_benchmark.cc
  kmeans_command_line_options.cc
)
set(HEADER_FILES
  kmeans_benchmark.h
  kmeans_command_line_options.h
)

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES} )

foreach(subdir ${SUBDIRECTORIES})
  add_subdirectory(${subdir})
endforeach(subdir)
