set(FOLDER_NAME bs_hip)
set(SUBDIRECTORIES)
set(SRC_FILES
        main.cpp
        bs_hip_benchmark.cpp
)
set(HEADER_FILES
  bs_hip_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (${HMARK_BUILD_HIP})
  add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
  set_source_files_properties(bs_hip_benchmark.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(${FOLDER_NAME}
      bs time_measurement command_line_option benchmark hip::host hip::device)
endif ()
