cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME be_hc)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  be_hc_benchmark.cc
)
set(HEADER_FILES 
  be_hc_benchmark.h
)
set(EXTRA_FILES 
)

if (COMPILE_HCC)
  if (OpenCV_FOUND)
    add_executable(${FOLDER_NAME} ${SRC_FILES})
    target_link_libraries(${FOLDER_NAME} 
      be time_measurement command_line_option benchmark ${OpenCV_LIBS})
    set_target_properties(${FOLDER_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})
  endif(OpenCV_FOUND)
endif (COMPILE_HCC)
