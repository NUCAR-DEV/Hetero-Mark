cmake_minimum_required(VERSION 3.2)
  
set(FOLDER_NAME ep_cuda)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  ep_cuda_benchmark.cu
)
set(HEADER_FILES
  ep_cuda_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (COMPILE_CUDA)
  cuda_add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME}
      ep time_measurement command_line_option benchmark)
endif (COMPILE_CUDA)

