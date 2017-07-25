cmake_minimum_required(VERSION 3.2)
  
set(FOLDER_NAME be_hip)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  be_hip_benchmark.cc
)
set(HEADER_FILES
  be_hip_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (COMPILE_HIP)
  if (OpenCV_FOUND)
	  add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
	  target_link_libraries(${FOLDER_NAME}
		  be time_measurement command_line_option benchmark ${OpenCV_LIBS})
  endif(OpenCV_FOUND)
endif (COMPILE_HIP)

