cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME be)
set(SUBDIRECTORIES 
  hc
)
set(SRC_FILES
  be_benchmark.cc
  be_command_line_options.cc
)
set(HEADER_FILES
  be_benchmark.h
  be_command_line_options.h
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
