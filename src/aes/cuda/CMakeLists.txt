set(FOLDER_NAME aes_cuda)
set(SUBDIRECTORIES)
set(SRC_FILES 
  main.cc
  aes_cuda_benchmark.cu
)
set(HEADER_FILES
  aes_cuda_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (${HMARK_BUILD_CUDA})
  add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME}
      aes time_measurement command_line_option benchmark)
endif ()

