if (${HMARK_BUILD_HIP})
    add_executable(ep_hip main.hip ep_hip_benchmark.hip ep_hip_benchmark.h)
    target_include_directories(ep_hip PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(ep_hip PUBLIC ep time_measurement command_line_option
            benchmark hip::host hip::device)
endif ()

