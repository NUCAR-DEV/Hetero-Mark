if (${HMARK_BUILD_HIP})
    add_executable(aes_hip main.hip aes_hip_benchmark.hip aes_hip_benchmark.h)
    target_include_directories(aes_hip PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(aes_hip PUBLIC aes time_measurement command_line_option
            benchmark hip::host hip::device)
endif ()
