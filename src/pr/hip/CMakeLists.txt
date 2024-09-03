set(FOLDER_NAME pr_hip)
set(SUBDIRECTORIES)
set(SRC_FILES
        main.hip
        pr_hip_benchmark.hip
)
set(HEADER_FILES
  pr_hip_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)


if (${HMARK_BUILD_HIP})
  add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME}
          pr time_measurement command_line_option benchmark hip::host hip::device)
endif ()
