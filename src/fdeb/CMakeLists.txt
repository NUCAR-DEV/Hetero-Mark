set(FOLDER_NAME fdeb)
set(SUBDIRECTORIES 
  hc
)
set(SRC_FILES
  fdeb_benchmark.cc
  fdeb_command_line_options.cc
)
set(HEADER_FILES
  fdeb_benchmark.h
  fdeb_command_line_options.h
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
