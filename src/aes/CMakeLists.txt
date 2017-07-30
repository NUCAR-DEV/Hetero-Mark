set(FOLDER_NAME aes)
set(SUBDIRECTORIES 
  cl12
  cl20
  hc
  cuda
  hip
)
set(SRC_FILES
  aes_benchmark.cc
  aes_command_line_options.cc
)
set(HEADER_FILES
  aes_benchmark.h
  aes_command_line_options.h
)

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
if (COMPILE_HCC)
  set_target_properties(${FOLDER_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})
endif (COMPILE_HCC)

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
