set(FOLDER_NAME be_hip)
set(SUBDIRECTORIES)
set(SRC_FILES
		main.hip be_hip_benchmark.hip
)
set(HEADER_FILES
  be_hip_benchmark.h
)
set(TEST_FILES)
set(EXTRA_FILES)

if (${HMARK_BUILD_HIP})
  if (${OpenCV_FOUND})
	  add_executable(${FOLDER_NAME} ${HEADER_FILES} ${SRC_FILES})
	  target_link_libraries(${FOLDER_NAME}
		  be time_measurement command_line_option benchmark hip::host hip::device ${OpenCV_LIBS})
  endif()
endif ()

