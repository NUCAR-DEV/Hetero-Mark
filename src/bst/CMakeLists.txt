set(FOLDER_NAME bst)
set(SUBDIRECTORIES 
  cuda
)
set(SRC_FILES
  bst_benchmark.cc
  bst_command_line_options.cc
)
set(HEADER_FILES
  bst_benchmark.h
  bst_command_line_options.h
)

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
