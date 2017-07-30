cmake_minimum_required(VERSION 3.2)
  
set(FOLDER_NAME ep_hip)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  ep_hip_benchmark.cu
)
set(HEADER_FILES
  ep_hip_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (COMPILE_HIP)
  HIP_ADD_EXECUTABLE(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME}
      ep time_measurement command_line_option benchmark)
endif (COMPILE_HIP)

