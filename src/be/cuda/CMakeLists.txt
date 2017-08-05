cmake_minimum_required(VERSION 3.2)
  
set(FOLDER_NAME be_cuda)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  be_cuda_benchmark.cu
)
set(HEADER_FILES
  be_cuda_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (COMPILE_CUDA)
  if (OpenCV_FOUND)
	  cuda_add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
	  target_link_libraries(${FOLDER_NAME}
		  be time_measurement command_line_option benchmark ${OpenCV_LIBS})
  endif(OpenCV_FOUND)
endif (COMPILE_CUDA)

