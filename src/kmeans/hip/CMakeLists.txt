cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME kmeans_hip)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  kmeans_hip_benchmark.cu
)
set(HEADER_FILES
  kmeans_hip_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (COMPILE_HIP)
  add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
  set_source_files_properties(kmeans_hip_benchmark.cu PROPERTIES LANGUAGE HIP)
  target_link_libraries(${FOLDER_NAME}
      kmeans time_measurement command_line_option benchmark)
endif (COMPILE_HIP)
