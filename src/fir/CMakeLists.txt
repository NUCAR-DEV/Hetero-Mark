set(FOLDER_NAME fir)
set(SUBDIRECTORIES 
  cl12
  cl20
  cuda
  hip
)
set(SRC_FILES
  fir_benchmark.cc
  fir_command_line_options.cc
)
set(HEADER_FILES
  fir_benchmark.h
  fir_command_line_options.h
)

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
