cmake_minimum_required(VERSION 3.2)
set(FOLDER_NAME pr)
set(SUBDIRECTORIES 
  cl12
  cl20
  hc
  cuda
  hip
)
set(SRC_FILES
  pr_benchmark.cc
  pr_command_line_options.cc
)
set(HEADER_FILES
  pr_benchmark.h
  pr_command_line_options.h
)

add_library(pr ${SRC_FILES} ${HEADER_FILES})

foreach(subdir ${SUBDIRECTORIES})
  add_subdirectory(${subdir})
endforeach(subdir )
