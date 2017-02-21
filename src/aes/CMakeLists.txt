set(FOLDER_NAME aes)
set(SUBDIRECTORIES 
  cl12
  cl20
  hsa
)
set(SRC_FILES
  aes_benchmark.cc
  aes_command_line_options.cc
)
set(HEADER_FILES
  aes_benchmark.h
  aes_command_line_options.h
)

add_library(aes ${SRC_FILES} ${HEADER_FILES})

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
